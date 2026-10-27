import type { components, paths } from './schema';
import type {
  PipelineCatalog,
  PipelineCatalogCacheEntry,
  PipelineCatalogResponse,
  PipelineDescriptor,
  PipelineSchemaResponse,
  PipelineParameterSchema,
  VideoAnnotatorFeatureFlags,
  VideoAnnotatorServerInfo,
  PipelineCapability,
  ExtrasInstallJob,
  ExtrasInstallTriggerResponse,
  VlmModelsResponse,
  ReadinessItem,
  PipelineReadiness,
  ExtrasGroupInfo,
  HuggingFaceTokenStatus
} from '@/types/pipelines';
import type { ServerLocations, SystemHealthResponse } from '@/types/system';
import type {
  BatchCancelResponse,
  BatchListResponse,
  BatchRetryResponse,
  BatchSummary,
} from '@/types/batches';
import type {
  IngestAccess,
  IngestBrowseResponse,
  IngestRequest,
  IngestResponse,
  Share,
} from '@/types/ingest';
import type { CurrentUser } from '@/types/api';
import type {
  AskDraftRequest,
  AskEntriesResponse,
  AskEntry,
  AskFrames,
  AskRunAnswers,
  AskRunRequest,
  AskTryRequest,
  ModelEndpoint,
  ModelServer,
  ModelServerAdd,
  ModelServerCheck,
  ModelServerCheckRequest,
  ModelServerPatch,
  ModelServerSettings,
  ModelServersResponse,
  PlayableCopy,
  StreamToken,
  SuggestShapeResponse,
  UploadedVideo,
} from '@/types/ask';
import type { JobResults } from '@/lib/jobOutcome';
import type { VideoNote, VideoNoteCreate, VideoNoteUpdate, VideoRuns } from '@/types/video';
import type { Preset, PresetCreateRequest, PresetListResponse, PresetUpdateRequest } from '@/types/presets';
import type {
  Activity,
  ResultValues,
  RunListResponse,
  RunSettings,
  VideosQuery,
  VideosResponse,
  VideosRunRequest,
  VideosRunResponse,
} from '@/types/videos';
import type { LibraryPrompt, PromptListResponse, PromptUpdateRequest } from '@/types/prompts';
import type {
  DatasetCreateRequest,
  DatasetListResponse,
  DatasetRunRequest,
  DatasetRunResponse,
  DatasetUpdateRequest,
  SavedDataset,
  ServerFolderScan,
  StoredVideosResponse,
} from '@/types/datasets';
import { APIError } from './handleError';
import { API_TOKEN_STORAGE_KEY, API_URL_STORAGE_KEY, defaultApiUrl, normalizeApiUrl } from '@/lib/apiConnection';

const optionalString = (value: unknown): string | undefined =>
  typeof value === 'string' && value ? value : undefined;

const optionalNumber = (value: unknown): number | undefined =>
  typeof value === 'number' ? value : undefined;

/** The pipeline guide (VideoAnnotator spec 026); nulls become undefined. */
function mapGuide(record: Record<string, unknown>): Partial<PipelineDescriptor> {
  const benchmarks = Array.isArray(record.benchmarks)
    ? (record.benchmarks as Record<string, unknown>[]).map((b) => ({
        name: String(b.name),
        measure: String(b.measure),
        figure: String(b.figure),
        source: String(b.source),
        note: optionalString(b.note),
      }))
    : undefined;
  const citations = Array.isArray(record.citations)
    ? (record.citations as Record<string, unknown>[]).map((c) => ({ key: String(c.key), bibtex: String(c.bibtex) }))
    : undefined;
  return {
    title: optionalString(record.title),
    summary: optionalString(record.summary),
    homepage: optionalString(record.homepage),
    benchmarks,
    qualityNote: optionalString(record.quality_note),
    citations,
  };
}

function mapTokenStatus(raw: Record<string, unknown>): HuggingFaceTokenStatus {
  const source = raw.source;
  const check = raw.check;
  return {
    set: raw.set === true,
    source: source === 'environment' || source === 'saved' ? source : null,
    envVar: optionalString(raw.env_var) ?? null,
    account: optionalString(raw.account) ?? null,
    check: check === 'ok' || check === 'invalid' || check === 'unknown' ? check : null,
  };
}

function mapParameter(raw: Record<string, unknown>): PipelineParameterSchema {
  const appliesWhen = raw.applies_when ?? raw.appliesWhen;
  return {
    ...(raw as unknown as PipelineParameterSchema),
    label: optionalString(raw.label),
    description: optionalString(raw.description),
    unit: optionalString(raw.unit),
    group: optionalString(raw.group),
    min: optionalNumber(raw.min),
    max: optionalNumber(raw.max),
    widget: optionalString(raw.widget),
    appliesWhen:
      appliesWhen && typeof appliesWhen === 'object' ? (appliesWhen as Record<string, string[]>) : undefined,
  };
}

const mapReadinessItems = (value: unknown): ReadinessItem[] =>
  Array.isArray(value)
    ? value
        .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
        .map((item) => ({
          kind: String(item.kind ?? ''),
          name: String(item.name ?? ''),
          message: String(item.message ?? ''),
          helpUrl: typeof item.help_url === 'string' ? item.help_url : null,
          approxMb: typeof item.approx_mb === 'number' ? item.approx_mb : null
        }))
    : [];

/** snake_case `readiness` (VideoAnnotator spec 011) -> PipelineReadiness; undefined from older servers. */
const mapReadiness = (value: unknown): PipelineReadiness | undefined => {
  if (!value || typeof value !== 'object') return undefined;
  const r = value as Record<string, unknown>;
  if (typeof r.state !== 'string') return undefined;
  return {
    state: r.state,
    nextAction: typeof r.next_action === 'string' ? r.next_action : 'none',
    extrasGroup: typeof r.extras_group === 'string' ? r.extras_group : null,
    installJobId: typeof r.install_job_id === 'string' ? r.install_job_id : null,
    blockers: mapReadinessItems(r.blockers),
    notes: mapReadinessItems(r.notes)
  };
};

export const getApiBaseUrl = (): string => {
  const saved = localStorage.getItem(API_URL_STORAGE_KEY);
  return saved === null ? defaultApiUrl() : normalizeApiUrl(saved);
};

const getApiToken = () => {
  const saved = localStorage.getItem(API_TOKEN_STORAGE_KEY);
  if (saved !== null) return saved;

  return import.meta.env.VITE_API_TOKEN || '';
};

/**
 * Whether any API token is configured (localStorage or env). Does not verify
 * the token is valid/admin — used only to decide whether to offer actions
 * that require auth (e.g. the extras-install button), which then handle a
 * 401/403 from the server as the definitive answer.
 */
export const hasConfiguredApiToken = (): boolean => getApiToken().trim() !== '';

/**
 * Validates if a token looks like a valid API key or JWT token
 * Valid formats:
 * - API Key: starts with 'va_' (e.g., 'va_xxxxxxxxxxxx')
 * - JWT: starts with 'eyJ' (e.g., 'eyJhbGciOiJIUzI1NiIs...')
 * - Other tokens: At least 8 characters and contains only valid characters
 * 
 * Rejects obviously invalid tokens like the old 'dev-token' placeholder, 'Bearer xyz', empty strings
 */
const isValidToken = (token: string): boolean => {
  if (!token || token.trim() === '') return false;

  const trimmed = token.trim();

  // Check for API key format (starts with 'va_')
  if (trimmed.startsWith('va_')) return true;

  // Check for JWT format (starts with 'eyJ')
  if (trimmed.startsWith('eyJ')) return true;

  // Reject known invalid patterns
  const invalidPatterns = ['dev-token', 'test-token', 'Bearer ', 'your-api-token'];
  if (invalidPatterns.some(pattern => trimmed.toLowerCase().includes(pattern.toLowerCase()))) {
    return false;
  }

  // Accept any token that's at least 8 characters and looks like a valid token
  // (alphanumeric, dashes, underscores, dots)
  if (trimmed.length >= 8 && /^[a-zA-Z0-9_.-]+$/.test(trimmed)) {
    return true;
  }

  // Reject everything else
  return false;
};

// Type definitions from OpenAPI schema
// Regenerate with scripts/gen_viewer_api_types.sh after changing the server's API.
export type JobResponse = components['schemas']['JobResponse'];

export interface BatchRerunResult {
  batch_id: string;
  rerun_of_batch: string;
  created: string[];
  skipped: { job_id: string; reason: string }[];
  /** Moved videos found in the chosen folder (spec 022). */
  relocated?: { job_id: string; from: string; to: string }[];
}
export type JobListResponse = components['schemas']['JobListResponse'];
export type PipelineResponse = components['schemas']['PipelineInfo'];
export type SubmitJobRequest = NonNullable<paths['/api/v1/jobs/']['post']['requestBody']>['content']['multipart/form-data'];

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
    this.baseURL = baseURL === undefined ? getApiBaseUrl() : normalizeApiUrl(baseURL);
    this.token = token || getApiToken();
  }

  // Update configuration dynamically
  /** undefined keeps a setting; '' is a value (this page's origin, or no token). */
  updateConfig(baseURL?: string, token?: string) {
    if (baseURL !== undefined) this.baseURL = normalizeApiUrl(baseURL);
    if (token !== undefined) this.token = token.trim();
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
    options: RequestInit = {},
    timeoutMs: number = 30000 // 30 second timeout (server takes 2-3s, browser needs buffer)
  ): Promise<T> {
    // Always get fresh values from localStorage in case they were updated
    this.baseURL = getApiBaseUrl().replace(/\/$/, '');
    this.token = getApiToken();

    const url = `${this.baseURL}${endpoint}`;

    const defaultHeaders: Record<string, string> = {};

    // Only add Authorization header if we have a valid token
    // This prevents JWT validation warnings on the server
    if (this.token && isValidToken(this.token)) {
      defaultHeaders['Authorization'] = `Bearer ${this.token}`;
      console.log('✅ Using valid token for request:', endpoint);
    } else if (this.token) {
      console.warn('⚠️ Token exists but failed validation:', {
        tokenPreview: this.token.substring(0, 10) + '...',
        tokenLength: this.token.length,
        startsWithVa: this.token.startsWith('va_'),
        startsWithEyJ: this.token.startsWith('eyJ')
      });
    }
    // If no valid token, make anonymous request (no Authorization header)

    // Only add Content-Type for requests with a body (POST/PUT/PATCH)
    // Don't add it for GET/DELETE as it triggers unnecessary CORS preflight
    if (options.body && !(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
      signal: controller.signal,
      mode: 'cors', // Explicitly enable CORS
      credentials: 'omit', // Don't send cookies (we use Authorization header)
    };

    console.log(`🌐 API Request: ${options.method || 'GET'} ${url}`, {
      headers: config.headers,
      mode: config.mode,
      credentials: config.credentials
    });

    try {
      const start = performance.now();
      const response = await fetch(url, config);
      const duration = Math.round(performance.now() - start);
      clearTimeout(timeoutId);

      console.log(`✅ API Response: ${response.status} ${response.statusText} (${duration}ms)`);

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        let errorData: unknown;

        try {
          errorData = (await response.json()) as unknown;
          const detail =
            errorData && typeof errorData === 'object'
              ? (errorData as Record<string, unknown>).detail
              : undefined;
          if (typeof detail === 'string') {
            errorMessage = detail;
          } else if (Array.isArray(detail)) {
            errorMessage = detail
              .map((entry) => {
                if (typeof entry === 'string') return entry;
                if (entry && typeof entry === 'object') {
                  const msg = (entry as Record<string, unknown>).msg;
                  if (typeof msg === 'string') return msg;
                }
                return JSON.stringify(entry);
              })
              .join(', ');
          }
        } catch {
          // If parsing error response fails, use default message
        }

        throw new APIError(errorMessage, response.status, response, errorData);
      }

      // Handle 204 No Content responses (like DELETE operations)
      if (response.status === 204) {
        return undefined as T;
      }

      // Handle responses that don't return JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        // No content type or not JSON - don't try to parse
        return undefined as T;
      }

      // Parse JSON response
      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof APIError) {
        throw error;
      }

      // Handle abort/timeout
      if (error instanceof Error && error.name === 'AbortError') {
        throw new APIError(
          `Request timeout: Server did not respond within ${timeoutMs / 1000} seconds. The server may be offline or unreachable.`,
          0
        );
      }

      // Network or other errors
      throw new APIError(
        `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        0
      );
    }
  }

  // detectServerInfo removed - not needed for core functionality

  async getServerInfo(forceRefresh = false): Promise<VideoAnnotatorServerInfo | null> {
    // Server info detection disabled - not essential for core functionality
    // Return cached value if available, otherwise null
    return this.serverInfoCache ?? null;
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

    const descriptors: PipelineDescriptor[] = pipelines.map((pipeline) => {
      const record = pipeline as unknown as Record<string, unknown>;
      const slug = record.slug;
      const name = record.name;
      const displayName = record.display_name;
      const description = record.description;
      const family = record.pipeline_family;
      const category = record.category;
      const version = record.version;
      const variant = record.variant;
      const modelName = record.model_name;
      const outputFormats = record.output_formats;
      const outputs = record.outputs;
      const defaultEnabled = record.default_enabled;
      const enabled = record.enabled;
      const available = record.available;
      const installHint = record.install_hint;

      const formatsFromOutputs = Array.isArray(outputs)
        ? outputs
            .map((output) => (output && typeof output === 'object' ? (output as Record<string, unknown>).format : null))
            .filter((value): value is string => typeof value === 'string')
        : [];

      const resolvedOutputFormats = Array.isArray(outputFormats)
        ? outputFormats.filter((value): value is string => typeof value === 'string')
        : formatsFromOutputs;

      return {
        id:
          (typeof slug === 'string' && slug) ||
          (typeof name === 'string' && name) ||
          'unknown',
        name:
          (typeof displayName === 'string' && displayName) ||
          (typeof name === 'string' && name) ||
          'unknown',
        description: typeof description === 'string' ? description : undefined,
        group:
          (typeof family === 'string' && family) ||
          (typeof category === 'string' && category) ||
          undefined,
        // The variant names the method, not a version; it's shown as the model.
        version: (typeof version === 'string' && version) || undefined,
        model:
          (typeof modelName === 'string' && modelName) ||
          (typeof variant === 'string' && variant) ||
          undefined,
        outputFormats: resolvedOutputFormats,
        defaultEnabled:
          typeof defaultEnabled === 'boolean'
            ? defaultEnabled
            : typeof enabled === 'boolean'
              ? enabled
              : true,
        capabilities: Array.isArray(record.capabilities)
          ? (record.capabilities as PipelineCapability[])
          : undefined,
        parameters: [],
        available: typeof available === 'boolean' ? available : undefined,
        installHint: typeof installHint === 'string' ? installHint : undefined,
        readiness: mapReadiness(record.readiness),
        ...mapGuide(record)
      };
    });

    return {
      pipelines: descriptors,
      source: 'legacy-list'
    };
  }

  private buildCatalogResponse(
    catalog: PipelineCatalog,
    server: VideoAnnotatorServerInfo | null,
    restartRequired: boolean
  ): PipelineCatalogResponse {
    const fallbackServer: VideoAnnotatorServerInfo =
      server ?? this.serverInfoCache ?? {
        version: 'unknown',
        features: this.featureFlags
      };

    return {
      catalog,
      server: fallbackServer,
      restartRequired
    };
  }

  async getPipelineCatalog(
    options: { forceRefresh?: boolean; includeUnavailable?: boolean } = {}
  ): Promise<PipelineCatalogResponse> {
    const { forceRefresh = false, includeUnavailable = false } = options;
    const now = Date.now();

    // Check cache first
    if (!forceRefresh && this.pipelineCatalogCache) {
      const age = now - this.pipelineCatalogCache.fetchedAt;
      if (age < this.pipelineCatalogTTL) {
        return this.buildCatalogResponse(
          this.pipelineCatalogCache.catalog,
          this.pipelineCatalogCache.server,
          this.pipelineCatalogCache.restartRequired
        );
      }
    }

    // Simple: just fetch pipelines, no server info needed
    const { pipelines: pipelineData, restartRequired } = await this.fetchPipelinesEnvelope(includeUnavailable);
    const catalog = this.mapLegacyPipelineResponse(pipelineData);

    // Use cached or default server info
    const serverInfo = this.serverInfoCache ?? {
      version: 'unknown',
      features: this.featureFlags
    };

    this.pipelineCatalogCache = {
      catalog,
      server: serverInfo,
      restartRequired,
      fetchedAt: now
    };

    return this.buildCatalogResponse(catalog, serverInfo, restartRequired);
  }

  async getPipelineSchema(pipelineId: string): Promise<PipelineSchemaResponse> {
    const serverInfo = await this.getServerInfo();
    const candidateEndpoints = [
      `/api/v1/pipelines/${pipelineId}/schema`,
      `/api/v1/pipelines/schema/${pipelineId}`
    ];

    for (const endpoint of candidateEndpoints) {
      try {
        const data = await this.request<unknown>(endpoint);
        if (data && typeof data === 'object') {
          const record = data as Record<string, unknown>;
          const parameters = record.parameters;

          if (!Array.isArray(parameters)) {
            continue;
          }

          this.featureFlags.pipelineSchemas = true;
          return {
            pipeline: (record.pipeline as PipelineDescriptor | undefined) ?? {
              id: pipelineId,
              name: (typeof record.name === 'string' ? record.name : pipelineId),
              description: typeof record.description === 'string' ? record.description : undefined,
              group: typeof record.group === 'string' ? record.group : undefined
            },
            parameters: (parameters as Record<string, unknown>[]).map(mapParameter)
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

  /**
   * The server process's identity (VideoAnnotator spec 011): `bootId` changes on
   * every start, which is how a client knows a restart finished. Both fields are
   * undefined on servers that predate it. Short timeout: this is polled while the
   * server is down.
   */
  async getBootIdentity(): Promise<{ bootId?: string; restartMode?: string }> {
    const response = await this.request<{ boot_id?: string; restart_mode?: string }>(
      '/health',
      {},
      3000
    );
    return { bootId: response?.boot_id, restartMode: response?.restart_mode };
  }

  /**
   * Installable extras groups with approximate download sizes (VideoAnnotator spec 011).
   * GET /api/v1/pipelines/extras. Returns null from servers that predate it (404).
   */
  async getExtrasGroups(): Promise<ExtrasGroupInfo[] | null> {
    try {
      const response = await this.request<{
        extras: {
          name: string;
          pipelines: string[];
          installed: boolean;
          approx_download_mb: number | null;
          includes_gpu_torch: boolean;
          install_job_id: string | null;
        }[];
      }>('/api/v1/pipelines/extras');
      return response.extras.map((g) => ({
        name: g.name,
        pipelines: g.pipelines,
        installed: g.installed,
        approxDownloadMb: g.approx_download_mb ?? null,
        includesGpuTorch: g.includes_gpu_torch === true,
        installJobId: g.install_job_id ?? null
      }));
    } catch (error) {
      if (error instanceof APIError && error.status === 404) return null;
      throw error;
    }
  }

  /**
   * Ask the server to restart itself (admin-only, VideoAnnotator spec 011).
   * POST /api/v1/system/restart -> 202; 409 with code RESTART_UNSUPPORTED,
   * JOBS_RUNNING (retry with force) or INSTALL_IN_PROGRESS; read those with
   * `apiErrorEnvelope`. 404 on servers that predate it.
   */
  async restartServer(force = false): Promise<{ bootId: string }> {
    const response = await this.request<{ restarting: boolean; boot_id: string }>(
      `/api/v1/system/restart${force ? '?force=true' : ''}`,
      { method: 'POST' }
    );
    return { bootId: response.boot_id };
  }

  /** Whether the server has a Hugging Face token (spec 031); never the token. */
  /** Where the server keeps results, model weights, logs and its database (UAT #16). */
  async getServerLocations(): Promise<ServerLocations> {
    return this.request<ServerLocations>('/api/v1/system/locations');
  }

  async getHuggingFaceToken(): Promise<HuggingFaceTokenStatus> {
    return mapTokenStatus(await this.request<Record<string, unknown>>('/api/v1/system/huggingface-token'));
  }

  /** Check and save a token on the server (admin). Rejected tokens throw an APIError (HF_TOKEN_REJECTED). */
  async saveHuggingFaceToken(token: string): Promise<HuggingFaceTokenStatus> {
    return mapTokenStatus(
      await this.request<Record<string, unknown>>('/api/v1/system/huggingface-token', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      })
    );
  }

  async removeHuggingFaceToken(): Promise<HuggingFaceTokenStatus> {
    return mapTokenStatus(
      await this.request<Record<string, unknown>>('/api/v1/system/huggingface-token', { method: 'DELETE' })
    );
  }

  async detailedHealth(): Promise<SystemHealthResponse> {
    return this.request('/api/v1/system/health');
  }

  async getSystemHealth(): Promise<SystemHealthResponse> {
    return this.detailedHealth();
  }

  // Job management endpoints
  async getJobs(
    page: number = 1,
    perPage: number = 20,
    options?: { batchId?: string; unbatchedOnly?: boolean }
  ): Promise<JobListResponse> {
    const params = new URLSearchParams({ page: String(page), per_page: String(perPage) });
    if (options?.batchId) params.set('batch_id', options.batchId);
    // Asking the server for "jobs in no batch" rather than filtering a page of
    // results client-side: otherwise a page filled with batched jobs hides
    // every ungrouped one behind it.
    if (options?.unbatchedOnly) params.set('unbatched_only', 'true');
    return this.request(`/api/v1/jobs?${params.toString()}`);
  }

  async getJob(jobId: string): Promise<JobResponse> {
    return this.request(`/api/v1/jobs/${jobId}`);
  }

  /** Saved pipeline presets (spec 007), shared by everyone on the server. */
  async listPresets(): Promise<PresetListResponse> {
    return this.request('/api/v1/presets');
  }

  async updatePreset(id: string, body: PresetUpdateRequest): Promise<Preset> {
    return this.request(`/api/v1/presets/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  }

  async deletePreset(id: string): Promise<void> {
    return this.request(`/api/v1/presets/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  async createPreset(body: PresetCreateRequest): Promise<Preset> {
    return this.request('/api/v1/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /**
   * Run a finished job again as a new job linked to it (spec 019). Omitted
   * settings keep the original's. 409 when it's still running or its video is gone.
   */
  async rerunJob(
    jobId: string,
    overrides: { selected_pipelines?: string[]; config?: Record<string, unknown> } = {}
  ): Promise<JobResponse> {
    return this.request(`/api/v1/jobs/${encodeURIComponent(jobId)}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(overrides),
    });
  }

  /**
   * Run a batch's finished jobs again as a new batch (spec 019). `check` only
   * reports what would be skipped; `relocateFolder` is where moved videos are
   * now, matched by name and size (spec 022).
   */
  async rerunBatch(
    batchId: string,
    overrides: { selected_pipelines?: string[]; config?: Record<string, unknown> } = {},
    options: { check?: boolean; relocateFolder?: string; recursive?: boolean } = {}
  ): Promise<BatchRerunResult> {
    const query = new URLSearchParams();
    if (options.check) query.set('check', 'true');
    if (options.relocateFolder) query.set('relocate_folder', options.relocateFolder);
    if (options.recursive) query.set('recursive', 'true');
    const suffix = query.toString() ? `?${query}` : '';
    return this.request(`/api/v1/batches/${encodeURIComponent(batchId)}/rerun${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(overrides),
    });
  }

  /** The prompt library (spec 020): every VLM prompt that ran, once each. */
  async listPrompts(params: { q?: string; model?: string; tag?: string; includeHidden?: boolean } = {}): Promise<PromptListResponse> {
    const query = new URLSearchParams();
    if (params.q) query.set('q', params.q);
    if (params.model) query.set('model', params.model);
    if (params.tag) query.set('tag', params.tag);
    if (params.includeHidden) query.set('include_hidden', 'true');
    return this.request(`/api/v1/prompts/?${query}`);
  }

  async updatePrompt(sha256: string, body: PromptUpdateRequest): Promise<LibraryPrompt> {
    return this.request(`/api/v1/prompts/${sha256}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  async deletePrompt(sha256: string): Promise<void> {
    return this.request(`/api/v1/prompts/${sha256}`, { method: 'DELETE' });
  }

  /** Saved datasets (specs 007, 018), shared by everyone on the server. */
  async listDatasets(): Promise<DatasetListResponse> {
    return this.request('/api/v1/datasets/');
  }

  async getDataset(id: string): Promise<SavedDataset> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}`);
  }

  /** Also the import path: POST an exported dataset as it is. 409 on a name the caller already uses. */
  async createDataset(body: DatasetCreateRequest): Promise<SavedDataset> {
    return this.request('/api/v1/datasets/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  async updateDataset(id: string, body: DatasetUpdateRequest): Promise<SavedDataset> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /** Which of a browser-uploaded dataset's videos the server still stores (from the jobs that ran on them). */
  async getStoredVideos(id: string): Promise<StoredVideosResponse> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}/stored-videos`);
  }

  /** One batch on the dataset's stored copies: no folder, no upload. */
  async runDataset(id: string, body: DatasetRunRequest): Promise<DatasetRunResponse> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  async deleteDataset(id: string): Promise<void> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  /** The videos a server-folder ingest would use, with sizes (admin, same machine). */
  async scanServerFolder(path: string, recursive: boolean): Promise<ServerFolderScan> {
    const query = new URLSearchParams({ path, recursive: String(recursive) });
    return this.request(`/api/v1/ingest/scan?${query}`);
  }

  /** Per-pipeline outcome of a job, including why a pipeline failed. */
  async getJobResults(jobId: string): Promise<JobResults> {
    return this.request(`/api/v1/jobs/${jobId}/results`);
  }

  async deleteJob(jobId: string): Promise<void> {
    return this.request(`/api/v1/jobs/${jobId}`, {
      method: 'DELETE',
    });
  }

  async submitJob(
    video: File,
    selectedPipelines?: string[],
    config?: Record<string, unknown>,
    batch?: { id: string; name?: string; datasetId?: string }
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

    // Spec 008: the batch id is the entire grouping mechanism — there is no
    // "create a batch" call. Every video in one submission carries the same id
    // (and name), and the server groups by it on read. Servers older than
    // v1.5.0 ignore these fields, so sending them is always safe.
    if (batch) {
      formData.append('batch_id', batch.id);
      if (batch.name) formData.append('batch_name', batch.name);
      if (batch.datasetId) formData.append('dataset_id', batch.datasetId);
    }

    return this.request('/api/v1/jobs', {
      method: 'POST',
      body: formData,
    });
  }

  // ==========================================================================
  // Batches (spec 008) — see src/types/batches.ts for why these are
  // hand-typed rather than derived from the generated schema.
  // ==========================================================================

  async getBatches(page: number = 1, perPage: number = 20): Promise<BatchListResponse> {
    return this.request(`/api/v1/batches?page=${page}&per_page=${perPage}`);
  }

  async getBatch(batchId: string): Promise<BatchSummary> {
    return this.request(`/api/v1/batches/${batchId}`);
  }

  async cancelBatch(batchId: string): Promise<BatchCancelResponse> {
    return this.request(`/api/v1/batches/${batchId}/cancel`, { method: 'POST' });
  }

  /** A run's results as one zip, without videos (spec 022). */
  async getBatchResultsZip(batchId: string): Promise<Blob> {
    return (await this.fetchRaw(`/api/v1/batches/${encodeURIComponent(batchId)}/results.zip`)).blob();
  }

  /** Delete a run: its jobs and results folder, never the videos (spec 022). */
  async deleteBatch(batchId: string): Promise<void> {
    return this.request(`/api/v1/batches/${encodeURIComponent(batchId)}`, { method: 'DELETE' });
  }

  /** Open a results folder in this computer's file manager (same machine only). */
  async openResultsFolder(path: string): Promise<void> {
    return this.request('/api/v1/results/open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
  }

  /** Run a failed run of one video again, in place. */
  async retryJob(jobId: string): Promise<JobResponse> {
    return this.request(`/api/v1/jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST' });
  }

  async retryBatch(batchId: string): Promise<BatchRetryResponse> {
    return this.request(`/api/v1/batches/${batchId}/retry`, { method: 'POST' });
  }

  // ==========================================================================
  // Server-side folder ingest — create jobs from videos already on the
  // server's disk, instead of uploading a corpus one file at a time.
  // Admin-only and local-callers-only server-side; see src/types/ingest.ts.
  // ==========================================================================

  /** What this browser may do with videos on the server's machine (spec 022). */
  async getIngestAccess(): Promise<IngestAccess> {
    return this.request('/api/v1/ingest/access');
  }

  /** Stop sharing a folder from the next start (spec 024; admin, same machine, launcher only). */
  async stopSharing(path: string): Promise<Share> {
    return this.request('/api/v1/ingest/shares/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
  }

  /** List server-side folders. Omit `path` to list the allowed roots. */
  async browseServerFolders(path?: string): Promise<IngestBrowseResponse> {
    const query = path ? `?path=${encodeURIComponent(path)}` : '';
    return this.request(`/api/v1/ingest/browse${query}`);
  }

  /** Turn every video in a server-side folder into one batch of jobs. */
  async ingestFolder(body: IngestRequest): Promise<IngestResponse> {
    return this.request('/api/v1/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  // Pipeline endpoints
  async getPipelines(): Promise<PipelineResponse[]> {
    return (await this.fetchPipelinesEnvelope()).pipelines;
  }

  private async fetchPipelinesEnvelope(
    includeUnavailable = false
  ): Promise<{ pipelines: PipelineResponse[]; restartRequired: boolean }> {
    const endpoint = includeUnavailable
      ? '/api/v1/pipelines?include_unavailable=true'
      : '/api/v1/pipelines';
    const response = await this.request<
      { pipelines: PipelineResponse[]; total?: number; restart_required?: boolean } | PipelineResponse[]
    >(endpoint);

    // Handle both legacy format (direct array) and new format (object with pipelines key)
    if (Array.isArray(response)) {
      return { pipelines: response, restartRequired: false };
    } else if (response && Array.isArray(response.pipelines)) {
      return { pipelines: response.pipelines, restartRequired: response.restart_required === true };
    } else {
      console.warn('Unexpected pipeline response format:', response);
      return { pipelines: [], restartRequired: false };
    }
  }

  /**
   * The authenticated caller's own identity, including admin scope.
   * GET /api/v1/auth/me - any authenticated caller can read this about
   * themselves (401 unauthenticated, never 403) - it's how the viewer knows
   * in advance whether an admin-gated action will succeed, instead of only
   * finding out via a bare 403. Absent on pre-v1.5.0 servers (404).
   */
  async getCurrentUser(): Promise<CurrentUser> {
    const response = await this.request<{
      id: string | number;
      username: string;
      email: string;
      is_admin: boolean;
    }>('/api/v1/auth/me');

    return {
      id: response.id,
      username: response.username,
      email: response.email,
      isAdmin: response.is_admin === true
    };
  }

  /**
   * Trigger a self-service install of a pipeline extras group.
   * POST /api/v1/pipelines/extras/{extra}/install (admin-only; 401/403/422 propagate as APIError)
   */
  async installPipelineExtras(extraName: string): Promise<ExtrasInstallTriggerResponse> {
    const response = await this.request<{ job_id: string; extra_name: string; status: string }>(
      `/api/v1/pipelines/extras/${encodeURIComponent(extraName)}/install`,
      { method: 'POST' }
    );

    return {
      jobId: response.job_id,
      extraName: response.extra_name,
      status: response.status as ExtrasInstallTriggerResponse['status']
    };
  }

  /**
   * Poll the status of an extras-install job.
   * GET /api/v1/pipelines/extras/install-jobs/{job_id} (404 once the job is unknown/expired)
   */
  async getExtrasInstallJob(jobId: string): Promise<ExtrasInstallJob> {
    const response = await this.request<{
      job_id: string;
      extra_name: string;
      status: string;
      created_at: string;
      started_at: string | null;
      finished_at: string | null;
      command_output: string | null;
      restart_required?: boolean;
      activation?: string | null;
      conflicting_distributions?: { name: string; old_version: string; new_version: string | null }[];
    }>(`/api/v1/pipelines/extras/install-jobs/${encodeURIComponent(jobId)}`);

    return {
      jobId: response.job_id,
      extraName: response.extra_name,
      status: response.status as ExtrasInstallJob['status'],
      createdAt: response.created_at,
      startedAt: response.started_at,
      finishedAt: response.finished_at,
      commandOutput: response.command_output,
      restartRequired: response.restart_required === true,
      activation: response.activation ?? null,
      conflictingDistributions: (response.conflicting_distributions ?? []).map((d) => ({
        name: d.name,
        oldVersion: d.old_version,
        newVersion: d.new_version
      }))
    };
  }

  /**
   * List vision-language models pulled on the configured Ollama server.
   * GET /api/v1/vlm/models — throws APIError(503, OLLAMA_UNREACHABLE) when
   * the server can't be reached at all; a reachable server with zero models
   * pulled returns normally with an empty `models` array (these are
   * deliberately distinct states — VideoAnnotator spec 009 FR-005).
   */
  async getVlmModels(server?: string): Promise<VlmModelsResponse> {
    const query = server && server !== 'default' ? `?server=${encodeURIComponent(server)}` : '';
    const response = await this.request<{
      base_url: string;
      models: string[];
      endpoint?: { host: string; class: 'local' | 'remote'; confirmed: boolean } | null;
      server?: string;
      kind?: 'ollama' | 'openai_compatible';
    }>(`/api/v1/vlm/models${query}`);
    return {
      baseUrl: response.base_url,
      models: response.models,
      endpoint: response.endpoint ? { baseUrl: response.base_url, ...response.endpoint } : undefined,
      server: response.server ?? 'default',
      kind: response.kind ?? 'ollama',
    };
  }

  // ------------------------------------------------------------------
  // Model servers (spec 033). No response carries a key.

  async listModelServers(): Promise<ModelServersResponse> {
    return this.request('/api/v1/model-servers');
  }

  async addModelServer(body: ModelServerAdd): Promise<ModelServer> {
    return this.request('/api/v1/model-servers', { method: 'POST', body: JSON.stringify(body) });
  }

  async updateModelServer(id: string, body: ModelServerPatch): Promise<ModelServer> {
    return this.request(`/api/v1/model-servers/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    });
  }

  async removeModelServer(id: string): Promise<void> {
    return this.request(`/api/v1/model-servers/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  async checkModelServer(body: ModelServerCheckRequest): Promise<ModelServerCheck> {
    return this.request('/api/v1/model-servers/check', { method: 'POST', body: JSON.stringify(body) });
  }

  async getModelServerSettings(): Promise<ModelServerSettings> {
    return this.request('/api/v1/model-servers/settings');
  }

  async setRemoteModelServers(on: boolean): Promise<ModelServerSettings> {
    return this.request('/api/v1/model-servers/settings', {
      method: 'PUT',
      body: JSON.stringify({ remote_model_servers: on }),
    });
  }

  async getAskSettings(): Promise<{ server: string }> {
    return this.request('/api/v1/ask/settings');
  }

  async setAskSettings(body: { server: string }): Promise<{ server: string }> {
    return this.request('/api/v1/ask/settings', { method: 'PUT', body: JSON.stringify(body) });
  }

  async suggestShape(body: {
    prompt: string;
    model?: string | null;
    base_url?: string | null;
    backend?: 'ollama' | 'openai_compatible';
  }): Promise<SuggestShapeResponse> {
    return this.request('/api/v1/ask/suggest-shape', { method: 'POST', body: JSON.stringify(body) });
  }

  // ------------------------------------------------------------------
  // Ask AI (spec 027) and videos the browser can play

  /** Upload a video once (stored by content); `onProgress` gets 0..1. */
  uploadVideo(file: File, onProgress?: (fraction: number) => void): Promise<UploadedVideo> {
    this.baseURL = getApiBaseUrl().replace(/\/$/, '');
    this.token = getApiToken();
    const form = new FormData();
    form.append('video', file, file.name);
    // XMLHttpRequest, not fetch: only it reports upload progress.
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseURL}/api/v1/videos`);
      if (this.token && isValidToken(this.token)) xhr.setRequestHeader('Authorization', `Bearer ${this.token}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body: unknown;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          body = undefined;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body as UploadedVideo);
        } else {
          const message = (body as { error?: { message?: string } } | undefined)?.error?.message ?? `HTTP ${xhr.status}`;
          reject(new APIError(message, xhr.status, undefined, body));
        }
      };
      xhr.onerror = () => reject(new APIError('Network error while uploading the video', 0));
      xhr.send(form);
    });
  }

  async streamToken(path: string): Promise<StreamToken> {
    return this.request('/api/v1/videos/stream-token', { method: 'POST', body: JSON.stringify({ path }) });
  }

  /** The `<video>` src for `path`: the token is the credential (a video element can't send the key). `copy`: its playable copy (D7). */
  streamUrl(path: string, token: string, copy = false): string {
    const base = getApiBaseUrl().replace(/\/$/, '');
    const params = new URLSearchParams({ path, token });
    if (copy) params.set('copy', 'true');
    return `${base}/api/v1/videos/stream?${params}`;
  }

  /** Whether `path` has a copy every browser can play (viewer design D7). */
  async getPlayableCopy(path: string): Promise<PlayableCopy> {
    return this.request(`/api/v1/videos/playable-copy?${new URLSearchParams({ path })}`);
  }

  /** Start making that copy on the server. */
  async makePlayableCopy(path: string): Promise<PlayableCopy> {
    return this.request('/api/v1/videos/playable-copy', { method: 'POST', body: JSON.stringify({ path }) });
  }

  /** Every run on one video, with its files, and the projects holding it (spec 028). */
  async getVideoRuns(path: string): Promise<VideoRuns> {
    return this.request(`/api/v1/videos/runs?${new URLSearchParams({ path })}`);
  }

  /** The videos the server knows, for one scope, with the Videos page's counts (spec 029). */
  async listVideos(query: VideosQuery = {}): Promise<VideosResponse> {
    const params = new URLSearchParams();
    if (query.project) params.set('project', query.project);
    if (query.folder) params.set('folder', query.folder);
    if (query.recursive) params.set('recursive', 'true');
    if (query.notInProject) params.set('not_in_project', 'true');
    if (query.run) params.set('run', query.run);
    if (query.q) params.set('q', query.q);
    if (query.sort) params.set('sort', query.sort);
    if (query.offset) params.set('offset', String(query.offset));
    params.set('limit', String(query.limit ?? 2000));
    return this.request(`/api/v1/videos?${params}`);
  }

  /** What each finished result found, as one value, by `<job_id>/<pipeline>` (UAT #10). */
  async getResultValues(results: { job_id: string; pipeline: string }[]): Promise<ResultValues> {
    const response = await this.request<{ values: ResultValues }>('/api/v1/videos/values', {
      method: 'POST',
      body: JSON.stringify({ results }),
    });
    return response.values;
  }

  /** One run on any videos the server can read (spec 029 R8). */
  async runVideos(body: VideosRunRequest): Promise<VideosRunResponse> {
    return this.request('/api/v1/videos/run', { method: 'POST', body: JSON.stringify(body) });
  }

  /** The settings New run starts from, and where they came from (spec 029 R6). */
  async getRunSettings(params: { project?: string | null; workflow?: string | null } = {}): Promise<RunSettings> {
    const query = new URLSearchParams();
    if (params.project) query.set('project', params.project);
    if (params.workflow) query.set('workflow', params.workflow);
    const suffix = query.toString() ? `?${query}` : '';
    return this.request(`/api/v1/videos/run-settings${suffix}`);
  }

  /** Runs with videos waiting or running, and those just finished (spec 029 R17). */
  async getActivity(): Promise<Activity> {
    return this.request('/api/v1/videos/activity');
  }

  /** Every run, those going on first, then newest first (UAT D2). */
  async listRuns(limit = 50, offset = 0): Promise<RunListResponse> {
    return this.request(`/api/v1/videos/all-runs?limit=${limit}&offset=${offset}`);
  }

  /** Add videos to a project by path; a project that followed its folder becomes a selection. */
  async addProjectVideos(id: string, paths: string[]): Promise<SavedDataset> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}/videos`, {
      method: 'POST',
      body: JSON.stringify({ paths }),
    });
  }

  /** Take videos out of a project; their files and results stay. */
  async removeProjectVideos(id: string, paths: string[]): Promise<SavedDataset> {
    return this.request(`/api/v1/datasets/${encodeURIComponent(id)}/videos`, {
      method: 'DELETE',
      body: JSON.stringify({ paths }),
    });
  }

  /** One result file, by the `download_url` the server listed for it (spec 028 R4). */
  async downloadResultFile(url: string, name: string): Promise<File> {
    const blob = await (await this.fetchRaw(url)).blob();
    return new File([blob], name);
  }

  /** A video's notes, by start (spec 028). */
  async listNotes(path: string): Promise<VideoNote[]> {
    const response = await this.request<{ notes: VideoNote[] }>(`/api/v1/videos/notes?${new URLSearchParams({ path })}`);
    return response.notes;
  }

  async createNote(body: VideoNoteCreate): Promise<VideoNote> {
    return this.request('/api/v1/videos/notes', { method: 'POST', body: JSON.stringify(body) });
  }

  async updateNote(id: string, body: VideoNoteUpdate): Promise<VideoNote> {
    return this.request(`/api/v1/videos/notes/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) });
  }

  async deleteNote(id: string): Promise<void> {
    await this.request(`/api/v1/videos/notes/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  async createTry(body: AskTryRequest): Promise<AskEntry> {
    return this.request('/api/v1/ask/tries', { method: 'POST', body: JSON.stringify(body) });
  }

  async createDraft(body: AskDraftRequest): Promise<AskEntry> {
    return this.request('/api/v1/ask/drafts', { method: 'POST', body: JSON.stringify(body) });
  }

  async getEntry(id: string): Promise<AskEntry> {
    return this.request(`/api/v1/ask/entries/${encodeURIComponent(id)}`);
  }

  async listEntries(
    params: { videoPath?: string; mine?: boolean; starred?: boolean; limit?: number; before?: string } = {}
  ): Promise<AskEntriesResponse> {
    const query = new URLSearchParams();
    if (params.videoPath) query.set('video_path', params.videoPath);
    if (params.mine) query.set('mine', 'true');
    if (params.starred !== undefined) query.set('starred', String(params.starred));
    if (params.limit) query.set('limit', String(params.limit));
    if (params.before) query.set('before', params.before);
    return this.request(`/api/v1/ask/entries?${query}`);
  }

  async updateEntry(id: string, body: { name?: string; starred?: boolean }): Promise<AskEntry> {
    return this.request(`/api/v1/ask/entries/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    });
  }

  async deleteEntry(id: string): Promise<void> {
    return this.request(`/api/v1/ask/entries/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  async createRun(body: AskRunRequest): Promise<AskEntry> {
    return this.request('/api/v1/ask/runs', { method: 'POST', body: JSON.stringify(body) });
  }

  async getRunAnswers(jobId: string, after = 0): Promise<AskRunAnswers> {
    return this.request(`/api/v1/ask/runs/${encodeURIComponent(jobId)}/answers?after=${after}`);
  }

  async getFrames(params: {
    path: string;
    t: number;
    interval: number;
    frames: number;
    spacing?: number | null;
    clip?: { length: number; fps: number } | null;
  }): Promise<AskFrames> {
    const query = new URLSearchParams({
      path: params.path,
      t: String(params.t),
      interval: String(params.interval),
      frames: String(params.frames),
    });
    if (params.spacing) query.set('spacing', String(params.spacing));
    if (params.clip) {
      query.set('clip_length', String(params.clip.length));
      query.set('clip_fps', String(params.clip.fps));
    }
    return this.request(`/api/v1/ask/frames?${query}`);
  }

  /** Where the model runs; works when the model server is unreachable too. */
  async getEndpoint(baseUrl?: string | null): Promise<ModelEndpoint> {
    const query = baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : '';
    const response = await this.request<{
      base_url: string;
      host: string;
      class: 'local' | 'remote';
      confirmed: boolean;
      sendable?: boolean;
      before_switch?: boolean;
    }>(`/api/v1/ask/endpoint${query}`);
    return {
      baseUrl: response.base_url,
      host: response.host,
      class: response.class,
      confirmed: response.confirmed,
      sendable: response.sendable,
      beforeSwitch: response.before_switch,
    };
  }

  async confirmEndpoint(baseUrl: string): Promise<void> {
    return this.request('/api/v1/ask/endpoints/confirm', {
      method: 'POST',
      body: JSON.stringify({ base_url: baseUrl }),
    });
  }

  /** A shaped question's answers, one column per field (spec 033 R10). */
  async exportEntryWide(id: string): Promise<ExportDownload> {
    const response = await this.fetchRaw(`/api/v1/ask/entries/${encodeURIComponent(id)}/export/wide`);
    return {
      blob: await response.blob(),
      filename: filenameFromDisposition(response.headers.get('Content-Disposition')),
      summary: null,
    };
  }

  /** A try's answers (or a run's) as the spec 025 table. */
  async exportEntryTable(id: string): Promise<ExportDownload> {
    const response = await this.fetchRaw(`/api/v1/ask/entries/${encodeURIComponent(id)}/export/table`);
    return {
      blob: await response.blob(),
      filename: filenameFromDisposition(response.headers.get('Content-Disposition')),
      summary: parseExportSummary(response.headers.get('X-VideoAnnotator-Export-Summary')),
    };
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
      // Refresh token from localStorage
      this.token = getApiToken();

      // Build headers for direct fetch calls
      const headers: Record<string, string> = {};
      if (this.token && isValidToken(this.token)) {
        headers['Authorization'] = `Bearer ${this.token}`;
      }

      // First, check if auth is required by examining health endpoint
      let authRequired = false;
      try {
        const healthResponse = await fetch(`${this.baseURL}/api/v1/system/health`);
        if (healthResponse.ok) {
          const healthData = await healthResponse.json();
          authRequired = healthData?.security?.auth_required === true;
        }
      } catch {
        // Health check failed, continue anyway
      }

      // Test with jobs endpoint - this actually requires proper auth when auth_required=true
      const jobsResponse = await fetch(`${this.baseURL}/api/v1/jobs?per_page=1`, { headers });

      // If jobs endpoint returns 401 or 403, auth is failing
      if (jobsResponse.status === 401 || jobsResponse.status === 403) {
        return { isValid: false, error: 'Authentication required or invalid token' };
      }

      // v1.3.0: When auth_required=true, server returns 404 for invalid/missing tokens
      // We check the response body to distinguish between error and empty list
      if (jobsResponse.status === 404) {
        try {
          const body = await jobsResponse.json();
          // If response has "detail" field, it's an error (auth failure or endpoint missing)
          if (body && typeof body === 'object' && 'detail' in body) {
             // If we are anonymous, this 404 likely means "Auth Required" (hidden resource)
             if (!this.token) {
                 return {
                    isValid: false,
                    error: 'Authentication required (Anonymous access not allowed)'
                 };
             }
             return {
                isValid: false,
                error: typeof body.detail === 'string' ? body.detail : 'Resource not found or authentication failed'
             };
          }
        } catch {
          // Failed to parse JSON, treat as error
          return {
            isValid: false,
            error: 'Authentication required or invalid token (404)'
          };
        }
      }

      // If jobs endpoint fails for other reasons (500, etc), check if server is reachable
      if (!jobsResponse.ok) {
        try {
          await this.healthCheck();
          // Health works but jobs doesn't - might be server issue
          return { isValid: false, error: `Server error: ${jobsResponse.status}` };
        } catch {
          // Both failed - server unreachable
          return { isValid: false, error: 'Cannot connect to server' };
        }
      }

      // Jobs endpoint returned success (200)
      // Consider token valid if we got here
      return { isValid: true, user: this.token ? 'Authenticated' : 'Anonymous' };
    } catch (error) {
      return {
        isValid: false,
        error: error instanceof Error ? error.message : 'Network error'
      };
    }
  }

  // =============================================================================
  // v1.3.0 ENHANCED API METHODS
  // =============================================================================

  /**
   * Cancel a running or queued job
   * POST /api/v1/jobs/{job_id}/cancel
   * 
   * @param jobId - Job ID to cancel
   * @param reason - Optional cancellation reason
   * @returns Job cancellation response
   * @throws APIError if cancellation fails
   */
  async cancelJob(
    jobId: string,
    reason?: string
  ): Promise<{
    job_id: string;
    status: 'cancelled' | 'cancelling';
    message: string;
    cancelled_at?: string;
  }> {
    const payload = reason ? { reason } : undefined;

    return this.request(`/api/v1/jobs/${jobId}/cancel`, {
      method: 'POST',
      body: payload ? JSON.stringify(payload) : undefined,
    });
  }

  /**
   * Download job artifacts as a ZIP file
   * GET /api/v1/jobs/{job_id}/artifacts
   * 
   * @param jobId - Job ID to download artifacts for
   * @returns Response object that can be used to stream the ZIP file
   */
  /** Fetch with this client's server and key, keeping the raw Response (files, videos). */
  private async fetchRaw(path: string, body?: unknown): Promise<Response> {
    this.baseURL = getApiBaseUrl().replace(/\/$/, '');
    this.token = getApiToken();
    const headers: Record<string, string> = {};
    if (this.token && isValidToken(this.token)) headers['Authorization'] = `Bearer ${this.token}`;
    const init: RequestInit =
      body === undefined
        ? { headers }
        : { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    const response = await fetch(`${this.baseURL}${path}`, init);
    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = undefined;
      }
      const message = (body as { error?: { message?: string } } | undefined)?.error?.message ?? `HTTP ${response.status}`;
      throw new APIError(message, response.status, response, body);
    }
    return response;
  }

  /**
   * Results out (spec 025): a job's, run's or dataset's results as one table,
   * ELAN files, or (via getMethods) a methods paragraph. The server's export
   * summary (what was covered and skipped) comes back with the file.
   */
  async exportResults(
    scope: ExportScope,
    kind: 'table' | 'elan',
    format: 'csv' | 'parquet' = 'csv'
  ): Promise<ExportDownload> {
    const response = isSetScope(scope)
      ? await this.fetchRaw(`/api/v1/videos/export/${kind}`, setBody(scope, kind === 'table' ? format : undefined))
      : await this.fetchRaw(`${exportBase(scope)}/export/${kind}${kind === 'table' ? `?format=${format}` : ''}`);
    return {
      blob: await response.blob(),
      filename: filenameFromDisposition(response.headers.get('Content-Disposition')),
      summary: parseExportSummary(response.headers.get('X-VideoAnnotator-Export-Summary')),
    };
  }

  /** A methods paragraph with BibTeX for a job's or run's results (spec 025). */
  async getMethods(scope: ExportScope): Promise<MethodsDescription> {
    if (isSetScope(scope)) return (await this.fetchRaw('/api/v1/videos/export/methods', setBody(scope, 'json'))).json();
    return (await this.fetchRaw(`${exportBase(scope)}/methods`)).json();
  }

  /** A pipeline's main output file for a job, as text (spec 021). */
  async getResultFileText(jobId: string, pipeline: string): Promise<string> {
    return (await this.fetchRaw(`/api/v1/jobs/${encodeURIComponent(jobId)}/results/files/${pipeline}`)).text();
  }

  /** The job's video as a blob URL (a <video> can't send the API key itself). Revoke it when done. */
  async getJobVideoUrl(jobId: string): Promise<string> {
    const blob = await (await this.fetchRaw(`/api/v1/jobs/${encodeURIComponent(jobId)}/video`)).blob();
    return URL.createObjectURL(blob);
  }

  /**
   * The job's results zip. Without its video unless `includeVideo` (spec 022:
   * the zip was otherwise one more copy of sensitive video).
   */
  async getJobArtifacts(jobId: string, options: { includeVideo?: boolean } = {}): Promise<Response> {
    // Always get fresh values from localStorage
    this.baseURL = getApiBaseUrl().replace(/\/$/, '');
    this.token = getApiToken();

    const query = options.includeVideo ? '?include_video=true' : '';
    const url = `${this.baseURL}/api/v1/jobs/${jobId}/artifacts${query}`;
    
    const headers: Record<string, string> = {};
    if (this.token && isValidToken(this.token)) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    // We use fetch directly here because we need the raw Response for streaming
    // and the request() wrapper parses JSON automatically
    const response = await fetch(url, {
      method: 'GET',
      headers,
      mode: 'cors',
      credentials: 'omit'
    });

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorData = (await response.json()) as unknown;
        const detail =
          errorData && typeof errorData === 'object'
            ? (errorData as Record<string, unknown>).detail
            : undefined;
        if (typeof detail === 'string') {
          errorMessage = detail;
        } else if (Array.isArray(detail)) {
          errorMessage = detail
            .map((entry) => {
              if (typeof entry === 'string') return entry;
              if (entry && typeof entry === 'object') {
                const msg = (entry as Record<string, unknown>).msg;
                if (typeof msg === 'string') return msg;
              }
              return JSON.stringify(entry);
            })
            .join(', ');
        }
      } catch {
        // Ignore JSON parse error
      }
      throw new APIError(errorMessage, response.status, response);
    }

    return response;
  }

  /**
   * Validate a full configuration object
   * POST /api/v1/config/validate
   * 
   * @param config - Configuration object to validate
   * @returns Validation result with errors/warnings
   * @throws APIError if validation request fails
   */
  async validateConfig(config: Record<string, unknown>): Promise<{
    valid: boolean;
    errors: Array<{
      field: string;
      message: string;
      severity: 'error' | 'warning' | 'info';
      error_code?: string;
      hint?: string;
      suggested_value?: unknown;
    }>;
    warnings: Array<{
      field: string;
      message: string;
      severity: 'error' | 'warning' | 'info';
      error_code?: string;
      hint?: string;
      suggested_value?: unknown;
    }>;
    validated_config?: Record<string, unknown>;
  }> {
    return this.request('/api/v1/config/validate', {
      method: 'POST',
      body: JSON.stringify({ config }),
    });
  }

  /**
   * Validate a pipeline-specific configuration
   * POST /api/v1/pipelines/{pipeline_name}/validate
   * 
   * @param pipelineName - Name of the pipeline
   * @param config - Configuration object to validate
   * @returns Validation result with errors/warnings
   * @throws APIError if validation request fails
   */
  async validatePipeline(
    pipelineName: string,
    config: Record<string, unknown>
  ): Promise<{
    valid: boolean;
    errors: Array<{
      field: string;
      message: string;
      severity: 'error' | 'warning' | 'info';
      error_code?: string;
      hint?: string;
      suggested_value?: unknown;
    }>;
    warnings: Array<{
      field: string;
      message: string;
      severity: 'error' | 'warning' | 'info';
      error_code?: string;
      hint?: string;
      suggested_value?: unknown;
    }>;
    validated_config?: Record<string, unknown>;
  }> {
    return this.request(`/api/v1/pipelines/${pipelineName}/validate`, {
      method: 'POST',
      body: JSON.stringify({ config }),
    });
  }

  /**
   * Get enhanced health status (v1.3.0+)
   * GET /api/v1/system/health
   * 
   * Returns enhanced health response with GPU status, worker info, diagnostics
   * Falls back gracefully to basic health response for v1.2.x servers
   * 
   * @returns Health response (enhanced if v1.3.0+, basic if v1.2.x)
   */
  async getEnhancedHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy' | string;
    version?: string;
    api_version?: string;
    videoannotator_version?: string;
    uptime_seconds?: number;
    gpu_status?: {
      available: boolean;
      device_name?: string;
      cuda_version?: string;
      memory_total?: number;
      memory_used?: number;
      memory_free?: number;
    };
    worker_info?: {
      active_jobs: number;
      queued_jobs: number;
      max_concurrent_jobs: number;
      worker_status: 'idle' | 'busy' | 'overloaded';
    };
    diagnostics?: {
      database: {
        status: 'healthy' | 'degraded' | 'unhealthy';
        message?: string;
      };
      storage: {
        status: 'healthy' | 'degraded' | 'unhealthy';
        disk_usage_percent?: number;
        message?: string;
      };
      ffmpeg: {
        available: boolean;
        version?: string;
      };
    };
    message?: string;
  }> {
    // Try enhanced endpoint first (v1.3.0)
    try {
      return await this.request('/api/v1/system/health');
    } catch (error) {
      // Fall back to basic health endpoint (v1.2.x)
      if (error instanceof APIError && error.status === 404) {
        return await this.request('/health');
      }
      throw error;
    }
  }
}

// Lazy singleton instance (avoids localStorage access at module load time)
let _apiClientInstance: APIClient | null = null;

/**
 * Get the singleton API client instance
 * Creates it on first access to avoid localStorage issues in tests
 */
export function getApiClient(): APIClient {
  if (!_apiClientInstance) {
    _apiClientInstance = new APIClient();
  }
  return _apiClientInstance;
}

// Export singleton as property for backward compatibility
// Methods are bound to the real client: called through the proxy, `this` was
// the proxy, so `this.token = ...` in updateConfig landed on the proxy's empty
// target and Settings' "Test Connection" kept testing the saved configuration.
export const apiClient = new Proxy({} as APIClient, {
  get(_target, prop) {
    const client = getApiClient();
    const value = Reflect.get(client, prop, client);
    return typeof value === 'function' ? value.bind(client) : value;
  },
  set(_target, prop, value) {
    return Reflect.set(getApiClient(), prop, value);
  },
});

// Export the class for custom instances
export { APIClient };



// --- Results out (spec 025) -------------------------------------------------

export type ExportScope =
  | { kind: 'job'; id: string }
  | { kind: 'batch'; id: string }
  | { kind: 'dataset'; id: string }
  // Spec 029: any videos, a project, or a run (a batch or a lone job), by POST.
  | { kind: 'videos'; paths: string[]; results?: ExportResults }
  | { kind: 'project'; id: string; results?: ExportResults }
  | { kind: 'run'; id: string };

/** Per video and pipeline its newest result, or every run's. */
export type ExportResults = 'latest' | 'all';

type SetScope = Extract<ExportScope, { kind: 'videos' | 'project' | 'run' }>;

function isSetScope(scope: ExportScope): scope is SetScope {
  return scope.kind === 'videos' || scope.kind === 'project' || scope.kind === 'run';
}

function setBody(scope: SetScope, format?: string): Record<string, unknown> {
  const results = scope.kind === 'run' ? 'all' : (scope.results ?? 'latest');
  const which =
    scope.kind === 'videos'
      ? { paths: scope.paths }
      : scope.kind === 'project'
        ? { project_id: scope.id }
        : { run_id: scope.id };
  return { ...which, results, ...(format ? { format } : {}) };
}

export interface ExportSummary {
  table_format_version: number;
  rows: number;
  videos: number;
  pipelines: string[];
  skipped: { run: string; video: string; file: string; reason: string }[];
  missing: { run: string; video: string; reason: string }[];
  notes: string[];
  left_out: string[];
  truncated: boolean;
}

export interface ExportDownload {
  blob: Blob;
  filename: string | null;
  summary: ExportSummary | null;
}

export interface MethodsDescription {
  prose: string;
  bibtex: string;
  revisions: string[];
  prompts: { sha256: string; text: string }[];
  summary: { videos: number; groups: number; not_recorded: string[] };
}

const EXPORT_PATHS: Record<Exclude<ExportScope, SetScope>['kind'], string> = {
  job: 'jobs',
  batch: 'batches',
  dataset: 'datasets',
};

function exportBase(scope: Exclude<ExportScope, SetScope>): string {
  return `/api/v1/${EXPORT_PATHS[scope.kind]}/${encodeURIComponent(scope.id)}`;
}

/** The file name a Content-Disposition header gives, preferring RFC 5987's UTF-8 form. */
export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1]);
    } catch {
      // fall through to the plain form
    }
  }
  const plain = /filename="([^"]*)"/i.exec(header);
  return plain ? plain[1] : null;
}

export function parseExportSummary(header: string | null): ExportSummary | null {
  if (!header) return null;
  try {
    return JSON.parse(header) as ExportSummary;
  } catch {
    return null;
  }
}
