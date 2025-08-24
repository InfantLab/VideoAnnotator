import type { paths } from './schema';

// API configuration with localStorage fallback
const getApiBaseUrl = () => {
  return localStorage.getItem('videoannotator_api_url') || 
         import.meta.env.VITE_API_BASE_URL || 
         'http://localhost:8000';
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

// API Error class
export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: Response
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// HTTP client with authentication and error handling
class APIClient {
  public baseURL: string;
  public token: string;

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
    return this.request('/api/v1/pipelines');
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
      
      // Try to get detailed token info if available
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
      } catch {
        // Debug endpoint might not exist
      }
      
      // Fallback: try an authenticated endpoint
      const response = await fetch(`${this.baseURL}/api/v1/jobs?per_page=1`, {
        headers: { 'Authorization': `Bearer ${this.token}` }
      });
      
      if (response.ok || response.status === 404) {
        return { isValid: true };
      } else if (response.status === 401) {
        return { isValid: false, error: 'Invalid or expired token' };
      } else {
        return { isValid: false, error: `Unexpected response: ${response.status}` };
      }
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

// Helper function to handle API errors in React components
export const handleAPIError = (error: unknown): string => {
  if (error instanceof APIError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
};