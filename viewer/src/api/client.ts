import type { paths } from './schema';
import type {
  PipelineCatalog,
  PipelineCatalogCacheEntry,
  PipelineCatalogResponse,
  PipelineDescriptor,
  PipelineSchemaResponse,
  PipelineParameterSchema,
  VideoAnnotatorFeatureFlags,
  VideoAnnotatorServerInfo,
  PipelineCapability
} from '@/types/pipelines';
import { APIError } from './handleError';

// API configuration with localStorage fallback
const getApiBaseUrl = () => {
  return localStorage.getItem('videoannotator_api_url') ||
    import.meta.env.VITE_API_BASE_URL ||
    'http://localhost:18011';
};

const getApiToken = () => {
  return localStorage.getItem('videoannotator_api_token') ||
    import.meta.env.VITE_API_TOKEN ||
    'dev-token';
};

// Type definitions from OpenAPI schema
export type JobResponse = paths['/api/v1/jobs']['get']['responses']['200']['content']['application/json']['jobs'][0];
export type JobListResponse = paths['/api/v1/jobs']['get']['responses']['200']['content']['application/json'];
export type PipelineResponse = paths['/api/v1/pipelines']['get']['responses']['200']['content']['application/json'][0];
export type SubmitJobRequest = paths['/api/v1/jobs']['post']['requestBody']['content']['multipart/form-data'];

// HTTP client with authentication and error handling
class APIClient {
  public baseURL: string;
  public token: string;
  private serverInfoCache: VideoAnnotatorServerInfo | null = null;
  private serverInfoPromise: Promise<VideoAnnotatorServerInfo | null> | null = null;
  private featureFlags: VideoAnnotatorFeatureFlags = {};
  private pipelineCatalogCache: PipelineCatalogCacheEntry | null = null;
  private readonly pipelineCatalogTTL = 5 * 60 * 1000; // 5 minutes

  constructor(baseURL?: string, token?: string) {
    this.baseURL = (baseURL || getApiBaseUrl()).replace(/\/$/, ''); // Remove trailing slash
    this.token = token || getApiToken();
  }

  // Update configuration dynamically
  updateConfig(baseURL?: string, token?: string) {
    if (baseURL) this.baseURL = baseURL.replace(/\/$/, '');
    if (token) this.token = token;
  }

  // Get current configuration
  getConfig() {
    return {
      baseURL: this.baseURL,
      token: this.token
    };
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Always get fresh values from localStorage in case they were updated
    this.baseURL = getApiBaseUrl().replace(/\/$/, '');
    this.token = getApiToken();

    const url = `${this.baseURL}${endpoint}`;

    const defaultHeaders: Record<string, string> = {
      'Authorization': `Bearer ${this.token}`,
    };

    // Only add Content-Type for non-FormData requests
    if (!(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = Array.isArray(errorData.detail)
              ? errorData.detail.map((e: any) => e.msg || e).join(', ')
              : errorData.detail;
          }
        } catch {
          // If parsing error response fails, use default message
        }

        throw new APIError(errorMessage, response.status, response);
      }

      // Handle responses that don't return JSON (like 204 No Content)
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        return {} as T;
      }
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }

      // Network or other errors
      throw new APIError(
        `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        0
      );
    }
  }

  private async detectServerInfo(): Promise<VideoAnnotatorServerInfo | null> {
    // Try endpoints in order from most likely to work to least likely
    // Based on server's actual OpenAPI spec at /openapi.json
    const candidateEndpoints = [
      '/api/v1/debug/server-info',  // ✅ Exists in server v1.2.0 OpenAPI spec
      '/api/v1/system/server-info', // ❌ Missing from server OpenAPI spec (404 expected)
      '/api/v1/system/info'         // ❌ Missing from server OpenAPI spec (404 expected)
    ];

    for (const endpoint of candidateEndpoints) {
      try {
        const data = await this.request<any>(endpoint);
        if (data) {
          const version =
            data.version ||
            data.server_version ||
            data.app_version ||
            data.git_sha ||
            'unknown';

          const features: VideoAnnotatorFeatureFlags = {
            ...this.featureFlags,
            pipelineCatalog: Boolean(data.features?.pipeline_catalog ?? data.pipeline_catalog),
            pipelineSchemas: Boolean(data.features?.pipeline_schemas ?? data.pipeline_schemas),
            pipelineHealth: Boolean(data.features?.pipeline_health ?? data.pipeline_health),
            jobSSE: Boolean(data.features?.job_sse ?? data.job_sse ?? true),
            artifactListing: Boolean(data.features?.artifact_listing ?? data.artifact_listing)
          };

          const capabilities: PipelineCapability[] | undefined = Array.isArray(data.capabilities)
            ? data.capabilities
            : undefined;

          return {
            version: String(version),
            build: data.build_id || data.build,
            commit: data.commit || data.git_sha,
            catalogVersion: data.catalog_version,
            lastUpdated: data.updated_at || data.last_updated,
            features,
            capabilities
          };
        }
      } catch (error) {
        if (error instanceof APIError && (error.status === 404 || error.status === 405)) {
          continue;
        }
        // Other failures should bubble up to let caller handle network issues
        throw error;
      }
    }

    return null;
  }

  async getServerInfo(forceRefresh = false): Promise<VideoAnnotatorServerInfo | null> {
    if (!forceRefresh && this.serverInfoCache) {
      return this.serverInfoCache;
    }

    // Prevent concurrent requests
    if (!forceRefresh && this.serverInfoPromise) {
      return this.serverInfoPromise;
    }

    this.serverInfoPromise = this.fetchServerInfo();
    const result = await this.serverInfoPromise;
    this.serverInfoPromise = null;

    return result;
  }

  private async fetchServerInfo(): Promise<VideoAnnotatorServerInfo | null> {
    try {
      const info = await this.detectServerInfo();
      if (info) {
        this.serverInfoCache = info;
        this.featureFlags = info.features ?? this.featureFlags;
        return info;
      }
    } catch (error) {
      // If we cannot fetch server info, ignore and rely on defaults
      console.warn('VideoAnnotator server info unavailable:', error);
    }

    return null;
  }

  private mapLegacyPipelineResponse(pipelines: PipelineResponse[]): PipelineCatalog {
    // Validate that pipelines is an array
    if (!Array.isArray(pipelines)) {
      console.warn('Expected pipelines to be an array, got:', typeof pipelines, pipelines);
      return {
        pipelines: [],
        source: 'legacy-list'
      };
    }

    const descriptors: PipelineDescriptor[] = pipelines.map((pipeline: any) => ({
      id: pipeline.slug || pipeline.name,
      name: pipeline.display_name || pipeline.name,
      description: pipeline.description,
      group: pipeline.pipeline_family || pipeline.category,
      version: pipeline.version || pipeline.variant || 'unknown',
      model: pipeline.model_name || pipeline.variant,
      outputFormats: pipeline.output_formats || (pipeline.outputs ? pipeline.outputs.map((o: any) => o.format) : []),
      defaultEnabled: pipeline.default_enabled ?? pipeline.enabled ?? true,
      capabilities: pipeline.capabilities as PipelineCapability[] | undefined,
      parameters: []
    }));

    return {
      pipelines: descriptors,
      source: 'legacy-list'
    };
  }

  private buildCatalogResponse(
    catalog: PipelineCatalog,
    server: VideoAnnotatorServerInfo | null
  ): PipelineCatalogResponse {
    const fallbackServer: VideoAnnotatorServerInfo =
      server ?? this.serverInfoCache ?? {
        version: 'unknown',
        features: this.featureFlags
      };

    return {
      catalog,
      server: fallbackServer
    };
  }

  async getPipelineCatalog(options: { forceRefresh?: boolean } = {}): Promise<PipelineCatalogResponse> {
    const { forceRefresh = false } = options;
    const now = Date.now();

    if (!forceRefresh && this.pipelineCatalogCache) {
      const age = now - this.pipelineCatalogCache.fetchedAt;
      if (age < this.pipelineCatalogTTL) {
        return this.buildCatalogResponse(
          this.pipelineCatalogCache.catalog,
          this.pipelineCatalogCache.server
        );
      }
    }

    const serverInfo = await this.getServerInfo(forceRefresh);

    // Use the actual server endpoint: /api/v1/pipelines/ (not /catalog)
    // This is the real endpoint that exists according to server OpenAPI spec
    try {
      const pipelineData = await this.getPipelines();
      const catalog = this.mapLegacyPipelineResponse(pipelineData);

      this.pipelineCatalogCache = {
        catalog,
        server: serverInfo ?? this.serverInfoCache ?? {
          version: 'unknown',
          features: this.featureFlags
        },
        fetchedAt: now
      };

      return this.buildCatalogResponse(catalog, serverInfo);
    } catch (error) {
      console.warn('Legacy pipeline endpoint also failed:', error);

      // Return empty catalog as final fallback
      const emptyCatalog: PipelineCatalog = {
        pipelines: [],
        source: 'fallback-empty'
      };

      this.pipelineCatalogCache = {
        catalog: emptyCatalog,
        server: serverInfo ?? this.serverInfoCache ?? {
          version: 'unknown',
          features: this.featureFlags
        },
        fetchedAt: now
      };

      return this.buildCatalogResponse(emptyCatalog, serverInfo);
    }
  }

  async getPipelineSchema(pipelineId: string): Promise<PipelineSchemaResponse> {
    const serverInfo = await this.getServerInfo();
    const candidateEndpoints = [
      `/api/v1/pipelines/${pipelineId}/schema`,
      `/api/v1/pipelines/schema/${pipelineId}`
    ];

    for (const endpoint of candidateEndpoints) {
      try {
        const data = await this.request<any>(endpoint);
        if (data && Array.isArray(data.parameters)) {
          this.featureFlags.pipelineSchemas = true;
          return {
            pipeline: data.pipeline ?? {
              id: pipelineId,
              name: data.name || pipelineId,
              description: data.description,
              group: data.group
            },
            parameters: data.parameters as PipelineParameterSchema[]
          };
        }
      } catch (error) {
        if (error instanceof APIError && (error.status === 404 || error.status === 405)) {
          continue;
        }
        throw error;
      }
    }

    // Fallback: try to find pipeline info from cached catalog
    const catalog = this.pipelineCatalogCache?.catalog ?? (await this.getPipelineCatalog()).catalog;
    const pipeline = catalog.pipelines.find((item) => item.id === pipelineId);

    return {
      pipeline: pipeline ?? {
        id: pipelineId,
        name: pipelineId
      },
      parameters: pipeline?.parameters ?? []
    };
  }

  clearPipelineCache() {
    this.pipelineCatalogCache = null;
  }

  clearServerInfoCache() {
    this.serverInfoCache = null;
  }

  getFeatureFlags(): VideoAnnotatorFeatureFlags {
    return { ...this.featureFlags };
  }

  // Health check endpoints
  async healthCheck(): Promise<{ status: string }> {
    return this.request('/health');
  }

  async detailedHealth(): Promise<any> {
    return this.request('/api/v1/system/health');
  }

  // Job management endpoints
  async getJobs(page: number = 1, perPage: number = 20): Promise<JobListResponse> {
    return this.request(`/api/v1/jobs?page=${page}&per_page=${perPage}`);
  }

  async getJob(jobId: string): Promise<JobResponse> {
    return this.request(`/api/v1/jobs/${jobId}`);
  }

  async submitJob(
    video: File,
    selectedPipelines?: string[],
    config?: Record<string, any>
  ): Promise<JobResponse> {
    const formData = new FormData();
    formData.append('video', video);

    // API expects comma-separated string, not JSON array
    if (selectedPipelines && selectedPipelines.length > 0) {
      formData.append('selected_pipelines', selectedPipelines.join(','));
    }

    if (config) {
      formData.append('config', JSON.stringify(config));
    }

    return this.request('/api/v1/jobs/', {
      method: 'POST',
      body: formData,
    });
  }

  // Pipeline endpoints
  async getPipelines(): Promise<PipelineResponse[]> {
    const response = await this.request<{ pipelines: PipelineResponse[]; total?: number }>('/api/v1/pipelines');

    // Handle both legacy format (direct array) and new format (object with pipelines key)
    if (Array.isArray(response)) {
      return response;
    } else if (response && Array.isArray(response.pipelines)) {
      return response.pipelines;
    } else {
      console.warn('Unexpected pipeline response format:', response);
      return [];
    }
  }

  // Server-Sent Events connection
  createEventSource(jobId?: string): EventSource {
    const url = jobId
      ? `${this.baseURL}/api/v1/events/stream?job_id=${jobId}&token=${this.token}`
      : `${this.baseURL}/api/v1/events/stream?token=${this.token}`;

    return new EventSource(url);
  }

  // Utility method to check if the API is reachable
  async isReachable(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  // Validate token and get user info
  async validateToken(): Promise<{
    isValid: boolean;
    user?: string;
    permissions?: string[];
    expiresAt?: string;
    error?: string;
  }> {
    try {
      // Try health check first
      await this.healthCheck();

      // Try to get detailed token info if available (optional debug endpoint)
      try {
        const response = await fetch(`${this.baseURL}/api/v1/debug/token-info`, {
          headers: { 'Authorization': `Bearer ${this.token}` }
        });

        if (response.ok) {
          const data = await response.json();
          return {
            isValid: true,
            user: data.token?.user_id,
            permissions: data.token?.permissions,
            expiresAt: data.token?.expires_at
          };
        }
        // If 401/404, silently fall through to alternative validation
      } catch {
        // Debug endpoint might not exist or require special permissions - this is normal
      }

      // Fallback: try multiple authenticated endpoints
      const candidateEndpoints = [
        '/api/v1/pipelines',     // This endpoint works according to test results
        '/api/v1/jobs?per_page=1' // Original fallback (might return 500)
      ];

      for (const endpoint of candidateEndpoints) {
        try {
          const response = await fetch(`${this.baseURL}${endpoint}`, {
            headers: { 'Authorization': `Bearer ${this.token}` }
          });

          if (response.ok || response.status === 404) {
            return { isValid: true };
          } else if (response.status === 401) {
            return { isValid: false, error: 'Invalid or expired token' };
          }
          // Continue to next endpoint if we get 500 or other errors
        } catch {
          // Continue to next endpoint on network errors
        }
      }

      // If all endpoints failed, assume invalid token
      return { isValid: false, error: 'Could not verify token with any endpoint' };
    } catch (error) {
      return {
        isValid: false,
        error: error instanceof Error ? error.message : 'Network error'
      };
    }
  }
}

// Export a singleton instance
export const apiClient = new APIClient();

// Export the class for custom instances
export { APIClient };
