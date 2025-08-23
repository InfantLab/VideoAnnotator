import type { paths } from './schema';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_TOKEN = import.meta.env.VITE_API_TOKEN || 'dev-token';

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
  private baseURL: string;
  private token: string;

  constructor(baseURL: string = API_BASE_URL, token: string = API_TOKEN) {
    this.baseURL = baseURL.replace(/\/$/, ''); // Remove trailing slash
    this.token = token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultHeaders: Record<string, string> = {
      'X-API-Key': this.token,
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
    
    if (selectedPipelines && selectedPipelines.length > 0) {
      formData.append('selected_pipelines', JSON.stringify(selectedPipelines));
    }
    
    if (config) {
      formData.append('config', JSON.stringify(config));
    }

    return this.request('/api/v1/jobs', {
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