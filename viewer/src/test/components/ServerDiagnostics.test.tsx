/**
 * T054: Component tests for ServerDiagnostics
 * Tests diagnostics display, auto-refresh, manual refresh, and error states
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ServerDiagnostics } from '@/components/ServerDiagnostics';

// Mock the API client
vi.mock('@/api/client', () => ({
    apiClient: {
        getEnhancedHealth: vi.fn(),
    },
    getApiClient: () => ({
        getEnhancedHealth: vi.fn(),
    }),
}));

import { apiClient } from '@/api/client';

describe('ServerDiagnostics', () => {
    let queryClient: QueryClient;
    let mockGetEnhancedHealth: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                },
            },
        });

        mockGetEnhancedHealth = apiClient.getEnhancedHealth as ReturnType<typeof vi.fn>;
        mockGetEnhancedHealth.mockReset();
    });

    afterEach(() => {
        vi.clearAllTimers();
    });

    const renderWithQueryClient = (ui: React.ReactElement) => {
        return render(
            <QueryClientProvider client={queryClient}>
                {ui}
            </QueryClientProvider>
        );
    };

    const mockHealthyResponse = {
        status: 'healthy' as const,
        version: '1.3.0',
        api_version: '1.3.0',
        uptime_seconds: 7200, // 2 hours
        gpu_status: {
            available: true,
            device_name: 'NVIDIA GeForce RTX 3080',
            cuda_version: '11.8',
            memory_total: 10737418240, // 10GB
            memory_used: 5368709120, // 5GB
            memory_free: 5368709120, // 5GB
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

    describe('basic rendering', () => {
        it('should render loading state initially', () => {
            mockGetEnhancedHealth.mockImplementation(() => new Promise(() => { })); // Never resolves

            renderWithQueryClient(<ServerDiagnostics />);

            expect(screen.getByText(/loading/i)).toBeInTheDocument();
        });

        it('should render error state when health check fails', async () => {
            mockGetEnhancedHealth.mockRejectedValueOnce(new Error('Connection failed'));

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/failed to load server diagnostics/i)).toBeInTheDocument();
            });
        });

        it('should render diagnostics when data loads successfully', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/server diagnostics/i)).toBeInTheDocument();
                expect(screen.getByText(/1.3.0/i)).toBeInTheDocument();
            });
        });
    });

    describe('server info display', () => {
        it('should display version and uptime', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/version/i)).toBeInTheDocument();
                expect(screen.getByText(/1.3.0/)).toBeInTheDocument();
                expect(screen.getByText(/uptime/i)).toBeInTheDocument();
                // Uptime should be formatted (e.g., "2 hours")
                expect(screen.getByText(/2 hours?/i)).toBeInTheDocument();
            });
        });

        it('should display status indicator', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/healthy/i)).toBeInTheDocument();
            });
        });
    });

    describe('GPU info display', () => {
        it('should display GPU information when available', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/gpu/i)).toBeInTheDocument();
                expect(screen.getByText(/NVIDIA GeForce RTX 3080/i)).toBeInTheDocument();
                expect(screen.getByText(/CUDA 11.8/i)).toBeInTheDocument();
                // Memory should be displayed as percentage
                expect(screen.getByText(/50%/)).toBeInTheDocument(); // 5GB / 10GB = 50%
            });
        });

        it('should show GPU unavailable message when GPU not present', async () => {
            const responseWithoutGPU = {
                ...mockHealthyResponse,
                gpu_status: {
                    available: false,
                },
            };

            mockGetEnhancedHealth.mockResolvedValueOnce(responseWithoutGPU);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/gpu not available/i)).toBeInTheDocument();
            });
        });

        it('should not show GPU section for v1.2.x servers', async () => {
            const v12Response = {
                status: 'healthy' as const,
                version: '1.2.1',
                api_version: '1.2.1',
            };

            mockGetEnhancedHealth.mockResolvedValueOnce(v12Response);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.queryByText(/gpu/i)).not.toBeInTheDocument();
            });
        });
    });

    describe('worker info display', () => {
        it('should display worker information', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/workers?/i)).toBeInTheDocument();
                expect(screen.getByText(/2/)).toBeInTheDocument(); // active jobs
                expect(screen.getByText(/3/)).toBeInTheDocument(); // queued jobs
                expect(screen.getByText(/4/)).toBeInTheDocument(); // max concurrent
            });
        });

        it('should color-code worker status', async () => {
            const busyResponse = { ...mockHealthyResponse };
            mockGetEnhancedHealth.mockResolvedValueOnce(busyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                const statusElement = screen.getByText(/busy/i);
                expect(statusElement).toBeInTheDocument();
                // Should have color indicator (e.g., yellow/amber for busy)
                expect(statusElement.closest('[class*="text-"]')).toBeTruthy();
            });
        });

        it('should show overloaded status in red', async () => {
            const overloadedResponse = {
                ...mockHealthyResponse,
                worker_info: {
                    active_jobs: 4,
                    queued_jobs: 10,
                    max_concurrent_jobs: 4,
                    worker_status: 'overloaded' as const,
                },
            };

            mockGetEnhancedHealth.mockResolvedValueOnce(overloadedResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                const statusElement = screen.getByText(/overloaded/i);
                expect(statusElement).toBeInTheDocument();
                // Should have red color indicator
                expect(statusElement.closest('[class*="text-red"]')).toBeTruthy();
            });
        });
    });

    describe('diagnostics display', () => {
        it('should display database status', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/database/i)).toBeInTheDocument();
                expect(screen.getByText(/all systems operational/i)).toBeInTheDocument();
            });
        });

        it('should display storage status with disk usage', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/storage/i)).toBeInTheDocument();
                expect(screen.getByText(/45%/)).toBeInTheDocument();
            });
        });

        it('should display FFmpeg status', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/ffmpeg/i)).toBeInTheDocument();
                expect(screen.getByText(/5.1.2/)).toBeInTheDocument();
            });
        });

        it('should show warning icon for degraded services', async () => {
            const degradedResponse = {
                ...mockHealthyResponse,
                diagnostics: {
                    ...mockHealthyResponse.diagnostics,
                    database: {
                        status: 'degraded' as const,
                        message: 'Slow query performance',
                    },
                },
            };

            mockGetEnhancedHealth.mockResolvedValueOnce(degradedResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/degraded/i)).toBeInTheDocument();
                expect(screen.getByText(/slow query performance/i)).toBeInTheDocument();
            });
        });
    });

    describe('manual refresh', () => {
        it('should have a refresh button', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
            });
        });

        it('should refetch data when refresh button clicked', async () => {
            const user = userEvent.setup();
            mockGetEnhancedHealth
                .mockResolvedValueOnce(mockHealthyResponse)
                .mockResolvedValueOnce({
                    ...mockHealthyResponse,
                    uptime_seconds: 7260, // 2 hours 1 minute
                });

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByText(/2 hours?/i)).toBeInTheDocument();
            });

            const refreshButton = screen.getByRole('button', { name: /refresh/i });
            await user.click(refreshButton);

            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
            });
        });

        it('should show loading state during manual refresh', async () => {
            const user = userEvent.setup();
            mockGetEnhancedHealth
                .mockResolvedValueOnce(mockHealthyResponse)
                .mockImplementation(() => new Promise(resolve => setTimeout(() => resolve(mockHealthyResponse), 100)));

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
            });

            const refreshButton = screen.getByRole('button', { name: /refresh/i });
            await user.click(refreshButton);

            // Button should be disabled or show loading state
            expect(refreshButton).toBeDisabled();
        });
    });

    describe('auto-refresh', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it('should auto-refresh every 30 seconds', async () => {
            mockGetEnhancedHealth
                .mockResolvedValueOnce(mockHealthyResponse)
                .mockResolvedValueOnce({
                    ...mockHealthyResponse,
                    uptime_seconds: 7230,
                })
                .mockResolvedValueOnce({
                    ...mockHealthyResponse,
                    uptime_seconds: 7260,
                });

            renderWithQueryClient(<ServerDiagnostics />);

            // Initial fetch
            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
            });

            // Fast forward 30 seconds
            vi.advanceTimersByTime(30000);

            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(2);
            });

            // Fast forward another 30 seconds
            vi.advanceTimersByTime(30000);

            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(3);
            });
        });

        it('should not auto-refresh when component unmounted', async () => {
            mockGetEnhancedHealth.mockResolvedValue(mockHealthyResponse);

            const { unmount } = renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
            });

            unmount();

            // Fast forward time - should not fetch again
            vi.advanceTimersByTime(60000);

            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
            });
        });
    });

    describe('stale data indicator', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it('should show stale data indicator after 2 minutes without refresh', async () => {
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.queryByText(/stale|outdated/i)).not.toBeInTheDocument();
            });

            // Fast forward 2 minutes
            vi.advanceTimersByTime(120000);

            await waitFor(() => {
                expect(screen.getByText(/stale|outdated/i)).toBeInTheDocument();
            });
        });

        it('should clear stale indicator after manual refresh', async () => {
            const user = userEvent.setup({ delay: null }); // No delay for fake timers
            mockGetEnhancedHealth
                .mockResolvedValueOnce(mockHealthyResponse)
                .mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
            });

            // Fast forward to make data stale
            vi.advanceTimersByTime(120000);

            await waitFor(() => {
                expect(screen.getByText(/stale|outdated/i)).toBeInTheDocument();
            });

            // Refresh
            const refreshButton = screen.getByRole('button', { name: /refresh/i });
            await user.click(refreshButton);

            await waitFor(() => {
                expect(screen.queryByText(/stale|outdated/i)).not.toBeInTheDocument();
            });
        });
    });

    describe('collapsible behavior', () => {
        it('should be collapsible', async () => {
            const user = userEvent.setup();
            mockGetEnhancedHealth.mockResolvedValueOnce(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            // Should have a trigger to expand/collapse
            const trigger = await screen.findByRole('button', { name: /server diagnostics/i });
            expect(trigger).toBeInTheDocument();
        });

        it('should stop auto-refresh when collapsed', async () => {
            vi.useFakeTimers();
            const user = userEvent.setup({ delay: null });
            mockGetEnhancedHealth.mockResolvedValue(mockHealthyResponse);

            renderWithQueryClient(<ServerDiagnostics />);

            await waitFor(() => {
                expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);
            });

            // Collapse the section
            const trigger = await screen.findByRole('button', { name: /server diagnostics/i });
            await user.click(trigger);

            // Fast forward - should not auto-refresh when collapsed
            vi.advanceTimersByTime(60000);

            expect(mockGetEnhancedHealth).toHaveBeenCalledTimes(1);

            vi.useRealTimers();
        });
    });
});
