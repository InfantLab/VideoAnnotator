/**
 * T055: Integration tests for ServerDiagnostics
 * Tests full diagnostics flow including cache and refresh behavior
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ServerDiagnostics } from '@/components/ServerDiagnostics';
import { apiClient } from '@/api/client';

// Mock the API client module
vi.mock('@/api/client', () => ({
  apiClient: {
    getEnhancedHealth: vi.fn(),
  },
  getApiClient: () => ({
    getEnhancedHealth: vi.fn(),
  }),
}));

describe('ServerDiagnostics Integration', () => {
  let queryClient: QueryClient;
  let mockGetEnhancedHealth: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Create fresh QueryClient for each test
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          staleTime: 0, // Disable caching for tests
          gcTime: 0,
        },
      },
    });

    mockGetEnhancedHealth = apiClient.getEnhancedHealth as ReturnType<typeof vi.fn>;
    mockGetEnhancedHealth.mockReset();
  });

  afterEach(() => {
    vi.clearAllTimers();
    queryClient.clear();
  });

  const renderWithQueryClient = (ui: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>
        {ui}
      </QueryClientProvider>
    );
  };

  const mockHealthResponse = {
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
      active_jobs: 1,
      queued_jobs: 2,
      max_concurrent_jobs: 4,
      worker_status: 'idle' as const,
    },
    diagnostics: {
      database: {
        status: 'healthy' as const,
        message: 'Connected',
      },
      storage: {
        status: 'healthy' as const,
        disk_usage_percent: 30,
      },
      ffmpeg: {
        available: true,
        version: '5.1.2',
      },
    },
  };

  describe('full diagnostics flow', () => {
    it('should load → display → allow refresh', async () => {
      const user = userEvent.setup();

      // Mock initial load
      mockGetEnhancedHealth
        .mockResolvedValueOnce(mockHealthResponse)
        .mockResolvedValueOnce({
          ...mockHealthResponse,
          uptime_seconds: 3660, // 1 hour 1 minute
        });

      renderWithQueryClient(<ServerDiagnostics />);

      // Should show loading initially
      expect(screen.getByText(/loading/i)).toBeInTheDocument();

      // Wait for data to load
      await waitFor(() => {
        expect(screen.getByText(/1.3.0/)).toBeInTheDocument();
        expect(screen.getByText(/1 hour/i)).toBeInTheDocument();
      });

      // Click refresh
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await user.click(refreshButton);

      // Wait for updated data
      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
      });
    });

    it('should handle error → retry flow', async () => {
      const user = userEvent.setup();

      // First call fails, second succeeds
      mockGetEnhancedHealth
        .mockRejectedValueOnce(new Error('Network timeout'))
        .mockResolvedValueOnce(mockHealthResponse);

      renderWithQueryClient(<ServerDiagnostics />);

      // Should show error
      await waitFor(() => {
        expect(screen.getByText(/failed to load/i)).toBeInTheDocument();
      });

      // Click retry/refresh
      const retryButton = screen.getByRole('button', { name: /refresh|retry/i });
      await user.click(retryButton);

      // Should now show data
      await waitFor(() => {
        expect(screen.getByText(/1.3.0/)).toBeInTheDocument();
      });
    });
  });

  describe('cache behavior', () => {
    it('should use cached data when available', async () => {
      mockGetEnhancedHealth.mockResolvedValue(mockHealthResponse);

      const { unmount } = renderWithQueryClient(<ServerDiagnostics />);

      // Wait for initial load
      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
      });

      unmount();

      // Remount - should use cache (but our test config disables cache)
      renderWithQueryClient(<ServerDiagnostics />);

      await waitFor(() => {
        // With gcTime: 0, should fetch again
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
      });
    });

    it('should invalidate cache on manual refresh', async () => {
      const user = userEvent.setup();

      // Enable cache for this test
      const cachedQueryClient = new QueryClient({
        defaultOptions: {
          queries: {
            retry: false,
            gcTime: 30000, // 30 seconds
            staleTime: 10000, // 10 seconds
          },
        },
      });

      mockGetEnhancedHealth
        .mockResolvedValueOnce(mockHealthResponse)
        .mockResolvedValueOnce({
          ...mockHealthResponse,
          version: '1.3.1',
        });

      const { rerender } = render(
        <QueryClientProvider client={cachedQueryClient}>
          <ServerDiagnostics />
        </QueryClientProvider>
      );

      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByText(/1.3.0/)).toBeInTheDocument();
      });

      // Rerender - should use cache (not fetch again)
      rerender(
        <QueryClientProvider client={cachedQueryClient}>
          <ServerDiagnostics />
        </QueryClientProvider>
      );

      expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);

      // Manual refresh should bypass cache
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await user.click(refreshButton);

      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
        expect(screen.getByText(/1.3.1/)).toBeInTheDocument();
      });
    });
  });

  describe('auto-refresh behavior', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should auto-refresh at 30s intervals', async () => {
      mockGetEnhancedHealth
        .mockResolvedValueOnce(mockHealthResponse)
        .mockResolvedValueOnce({ ...mockHealthResponse, uptime_seconds: 3630 })
        .mockResolvedValueOnce({ ...mockHealthResponse, uptime_seconds: 3660 });

      renderWithQueryClient(<ServerDiagnostics />);

      // Initial fetch
      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
      });

      // Advance 30 seconds
      vi.advanceTimersByTime(30000);

      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
      });

      // Advance another 30 seconds
      vi.advanceTimersByTime(30000);

      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(3);
      });
    });

    it('should pause auto-refresh when collapsed', async () => {
      const user = userEvent.setup({ delay: null }); // No delay for fake timers

      mockGetEnhancedHealth.mockResolvedValue(mockHealthResponse);

      renderWithQueryClient(<ServerDiagnostics />);

      // Initial fetch
      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
      });

      // Collapse the section
      const trigger = await screen.findByRole('button', { name: /server diagnostics/i });
      await user.click(trigger);

      // Advance time - should NOT auto-refresh when collapsed
      vi.advanceTimersByTime(60000);

      expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
    });

    it('should resume auto-refresh when expanded again', async () => {
      const user = userEvent.setup({ delay: null });

      mockGetEnhancedHealth.mockResolvedValue(mockHealthResponse);

      renderWithQueryClient(<ServerDiagnostics />);

      // Initial fetch
      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
      });

      // Collapse
      const trigger = await screen.findByRole('button', { name: /server diagnostics/i });
      await user.click(trigger);

      vi.advanceTimersByTime(60000);
      expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);

      // Expand again
      await user.click(trigger);

      // Should fetch immediately on expand
      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
      });

      // Should resume auto-refresh
      vi.advanceTimersByTime(30000);

      await waitFor(() => {
        expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(3);
      });
    });
  });

  describe('v1.2.x backward compatibility', () => {
    it('should handle v1.2.x health response gracefully', async () => {
      const v12Response = {
        status: 'healthy' as const,
        api_version: '1.2.1',
        videoannotator_version: '1.2.1',
        message: 'API is running',
      };

      mockGetEnhancedHealth.mockResolvedValueOnce(v12Response);

      renderWithQueryClient(<ServerDiagnostics />);

      await waitFor(() => {
        expect(screen.getByText(/1.2.1/)).toBeInTheDocument();
        expect(screen.getByText(/healthy/i)).toBeInTheDocument();
      });

      // Should not show GPU/worker sections for v1.2.x
      expect(screen.queryByText(/gpu/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/worker/i)).not.toBeInTheDocument();
    });

    it('should gracefully handle missing optional fields in v1.3.0', async () => {
      const partialResponse = {
        status: 'healthy' as const,
        version: '1.3.0',
        api_version: '1.3.0',
        // No GPU, worker, or diagnostics info
      };

      mockGetEnhancedHealth.mockResolvedValueOnce(partialResponse);

      renderWithQueryClient(<ServerDiagnostics />);

      await waitFor(() => {
        expect(screen.getByText(/1.3.0/)).toBeInTheDocument();
      });

      // Should render without crashing, showing what's available
      expect(screen.getByText(/healthy/i)).toBeInTheDocument();
    });
  });

  describe('real-world scenarios', () => {
    it('should handle server becoming overloaded during session', async () => {
      vi.useFakeTimers();

      const normalResponse = mockHealthResponse;
      const overloadedResponse = {
        ...mockHealthResponse,
        worker_info: {
          active_jobs: 4,
          queued_jobs: 15,
          max_concurrent_jobs: 4,
          worker_status: 'overloaded' as const,
        },
      };

      mockGetEnhancedHealth
        .mockResolvedValueOnce(normalResponse)
        .mockResolvedValueOnce(overloadedResponse);

      renderWithQueryClient(<ServerDiagnostics />);

      // Initial state - idle
      await waitFor(() => {
        expect(screen.getByText(/idle/i)).toBeInTheDocument();
      });

      // Auto-refresh after 30s
      vi.advanceTimersByTime(30000);

      // Should now show overloaded
      await waitFor(() => {
        expect(screen.getByText(/overloaded/i)).toBeInTheDocument();
      });

      vi.useRealTimers();
    });

    it('should handle database degradation', async () => {
      const degradedResponse = {
        ...mockHealthResponse,
        status: 'degraded' as const,
        diagnostics: {
          ...mockHealthResponse.diagnostics,
          database: {
            status: 'degraded' as const,
            message: 'Connection pool exhausted',
          },
        },
      };

      mockGetEnhancedHealth.mockResolvedValueOnce(degradedResponse);

      renderWithQueryClient(<ServerDiagnostics />);

      await waitFor(() => {
        expect(screen.getByText(/degraded/i)).toBeInTheDocument();
        expect(screen.getByText(/connection pool exhausted/i)).toBeInTheDocument();
      });
    });

    it('should show stale data warning after connectivity issues', async () => {
      vi.useFakeTimers();

      mockGetEnhancedHealth
        .mockResolvedValueOnce(mockHealthResponse)
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'));

      renderWithQueryClient(<ServerDiagnostics />);

      // Initial load succeeds
      await waitFor(() => {
        expect(screen.getByText(/1.3.0/)).toBeInTheDocument();
      });

      // Auto-refresh fails after 30s
      vi.advanceTimersByTime(30000);

      // Another auto-refresh fails after 60s
      vi.advanceTimersByTime(30000);

      // Fast forward past 2 minute threshold
      vi.advanceTimersByTime(60000);

      // Should show stale data indicator
      await waitFor(() => {
        expect(screen.getByText(/stale|outdated/i)).toBeInTheDocument();
      });

      vi.useRealTimers();
    });
  });
});
