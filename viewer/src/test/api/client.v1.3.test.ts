// Unit tests for v1.3.0 API client methods
// Tests for cancelJob, validateConfig, validatePipeline, getEnhancedHealth

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { APIClient } from '@/api/client';

describe('APIClient v1.3.0 methods', () => {
  let client: APIClient;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Mock localStorage.getItem to return null (triggers fallback to env vars)
    if (global.localStorage && global.localStorage.getItem) {
      (global.localStorage.getItem as any) = vi.fn(() => null);
    }

    // Reset mocks
    mockFetch = vi.fn();
    global.fetch = mockFetch as any;

    // Create fresh client instance
    client = new APIClient('http://localhost:18011', 'test-token');
  });

  describe('cancelJob', () => {
    it('should cancel a job successfully', async () => {
      const mockResponse = {
        job_id: 'job123',
        status: 'cancelled' as const,
        message: 'Job cancelled successfully',
        cancelled_at: '2025-10-27T10:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const result = await client.cancelJob('job123');

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:18011/api/v1/jobs/job123/cancel',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            Authorization: expect.stringMatching(/^Bearer /),
          }),
        })
      );
    });

    it('should handle cancellation with reason', async () => {
      const mockResponse = {
        job_id: 'job456',
        status: 'cancelling' as const,
        message: 'Cancellation in progress',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const result = await client.cancelJob('job456', 'User requested');

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:18011/api/v1/jobs/job456/cancel',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ reason: 'User requested' }),
        })
      );
    });

    it('should handle 400 error for already cancelled job', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          error: 'Job already cancelled',
          error_code: 'JOB_ALREADY_CANCELLED',
        }),
      });

      await expect(client.cancelJob('job789')).rejects.toThrow();
    });

    it('should handle 404 error for non-existent job', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          error: 'Job not found',
          error_code: 'JOB_NOT_FOUND',
        }),
      });

      await expect(client.cancelJob('nonexistent')).rejects.toThrow();
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(client.cancelJob('job999')).rejects.toThrow('Network error');
    });
  });

  describe('validateConfig', () => {
    it('should validate a valid config', async () => {
      const mockResponse = {
        valid: true,
        errors: [],
        warnings: [],
        validated_config: {
          pipeline: 'pose_detection',
          confidence_threshold: 0.5,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const config = {
        pipeline: 'pose_detection',
        confidence_threshold: 0.5,
      };

      const result = await client.validateConfig(config);

      expect(result).toEqual(mockResponse);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should return errors for invalid config', async () => {
      const mockResponse = {
        valid: false,
        errors: [
          {
            field: 'confidence_threshold',
            message: 'Value must be between 0 and 1',
            severity: 'error' as const,
            error_code: 'VALUE_OUT_OF_RANGE',
            hint: 'Try a value like 0.5',
          },
        ],
        warnings: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const config = {
        confidence_threshold: 1.5, // Invalid value
      };

      const result = await client.validateConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].field).toBe('confidence_threshold');
      expect(result.errors[0].hint).toBe('Try a value like 0.5');
    });

    it('should return warnings for suboptimal config', async () => {
      const mockResponse = {
        valid: true,
        errors: [],
        warnings: [
          {
            field: 'confidence_threshold',
            message: 'Low threshold may result in false positives',
            severity: 'warning' as const,
            error_code: 'SUBOPTIMAL_VALUE',
            suggested_value: 0.5,
          },
        ],
        validated_config: {
          confidence_threshold: 0.1,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const config = {
        confidence_threshold: 0.1,
      };

      const result = await client.validateConfig(config);

      expect(result.valid).toBe(true);
      expect(result.warnings).toHaveLength(1);
      expect(result.warnings[0].suggested_value).toBe(0.5);
    });
  });

  describe('validatePipeline', () => {
    it('should validate pipeline-specific config', async () => {
      const mockResponse = {
        valid: true,
        errors: [],
        warnings: [],
        validated_config: {
          model: 'yolov8n',
          confidence: 0.5,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const config = {
        model: 'yolov8n',
        confidence: 0.5,
      };

      const result = await client.validatePipeline('pose_detection', config);

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:18011/api/v1/pipelines/pose_detection/validate',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ config }),
        })
      );
    });
  });

  describe('getEnhancedHealth', () => {
    it('should return enhanced health response from v1.3.0 server', async () => {
      const mockResponse = {
        status: 'healthy' as const,
        version: '1.3.0',
        api_version: '1.3.0',
        uptime_seconds: 3600,
        gpu_status: {
          available: true,
          device_name: 'NVIDIA GeForce RTX 3080',
          cuda_version: '11.8',
          memory_total: 10737418240,
          memory_used: 5368709120,
          memory_free: 5368709120,
        },
        worker_info: {
          active_jobs: 2,
          queued_jobs: 3,
          max_concurrent_jobs: 4,
          worker_status: 'busy' as const,
        },
        diagnostics: {
          database: {
            status: 'healthy' as const,
            message: 'All systems operational',
          },
          storage: {
            status: 'healthy' as const,
            disk_usage_percent: 45,
          },
          ffmpeg: {
            available: true,
            version: '5.1.2',
          },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockResponse,
      });

      const result = await client.getEnhancedHealth();

      expect(result).toEqual(mockResponse);
      expect(result.uptime_seconds).toBe(3600);
      expect(result.gpu_status?.available).toBe(true);
      expect(result.worker_info?.worker_status).toBe('busy');
    });

    it('should fall back to /health for v1.2.x server', async () => {
      const mockEnhancedResponse = {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ error: 'Endpoint not found' }),
      };

      const mockBasicResponse = {
        status: 'healthy',
        api_version: '1.2.1',
        videoannotator_version: '1.2.1',
        message: 'API is running',
      };

      mockFetch
        .mockResolvedValueOnce(mockEnhancedResponse)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          headers: new Headers({ 'content-type': 'application/json' }),
          json: async () => mockBasicResponse,
        });

      const result = await client.getEnhancedHealth();

      expect(result).toEqual(mockBasicResponse);
      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch).toHaveBeenNthCalledWith(
        1,
        'http://localhost:18011/api/v1/system/health',
        expect.anything()
      );
      expect(mockFetch).toHaveBeenNthCalledWith(
        2,
        'http://localhost:18011/health',
        expect.anything()
      );
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Connection refused'));

      await expect(client.getEnhancedHealth()).rejects.toThrow('Connection refused');
    });
  });
});
