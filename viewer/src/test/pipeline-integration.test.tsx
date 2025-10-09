/**
 * @file Tests for pipeline data hooks and context
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { usePipelineData } from '@/hooks/usePipelineData';
import { PipelineProvider, usePipelineContext } from '@/contexts/PipelineContext';
import { apiClient } from '@/api/client';

// Mock localStorage
const localStorageMock = {
    getItem: vi.fn(() => null),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn(() => null),
};

Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
    writable: true,
});

// Mock the API client
vi.mock('@/api/client', () => ({
    apiClient: {
        getPipelineCatalog: vi.fn(),
        getServerInfo: vi.fn(),
        clearPipelineCache: vi.fn(),
        clearServerInfoCache: vi.fn(),
    }
}));

const createWrapper = () => {
    const queryClient = new QueryClient({
        defaultOptions: {
            queries: { retry: false },
            mutations: { retry: false },
        },
    });

    return ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>
            <PipelineProvider>
                {children}
            </PipelineProvider>
        </QueryClientProvider>
    );
};

describe('Pipeline Data Hooks', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should provide pipeline context with default values', () => {
        const mockCatalogResponse = {
            catalog: {
                pipelines: [
                    {
                        id: 'face_analysis',
                        name: 'Face Analysis',
                        group: 'Face',
                        description: 'OpenFace3 face analysis',
                        defaultEnabled: true,
                        capabilities: [
                            { feature: 'landmarks_2d', enabled: true },
                            { feature: 'emotions', enabled: true }
                        ]
                    }
                ]
            },
            server: {
                version: '1.2.1',
                features: {
                    pipelineCatalog: true,
                    pipelineSchemas: true
                }
            }
        };

        vi.mocked(apiClient.getPipelineCatalog).mockResolvedValue(mockCatalogResponse);
        vi.mocked(apiClient.getServerInfo).mockResolvedValue(mockCatalogResponse.server);

        const { result } = renderHook(() => usePipelineContext(), {
            wrapper: createWrapper(),
        });

        expect(result.current).toBeDefined();
        expect(result.current.isLoading).toBe(true); // Initially loading
        expect(typeof result.current.isPipelineAvailable).toBe('function');
        expect(typeof result.current.getPipeline).toBe('function');
    });

    it('should check pipeline availability correctly', async () => {
        const mockCatalogResponse = {
            catalog: {
                pipelines: [
                    {
                        id: 'face_analysis',
                        name: 'Face Analysis',
                        group: 'Face',
                        description: 'OpenFace3 face analysis',
                        defaultEnabled: true,
                        capabilities: []
                    }
                ]
            },
            server: {
                version: '1.2.1',
                features: {
                    pipelineCatalog: true
                }
            }
        };

        vi.mocked(apiClient.getPipelineCatalog).mockResolvedValue(mockCatalogResponse);
        vi.mocked(apiClient.getServerInfo).mockResolvedValue(mockCatalogResponse.server);

        const { result } = renderHook(() => usePipelineData(), {
            wrapper: createWrapper(),
        });

        // Wait for initial load to complete
        await new Promise(resolve => setTimeout(resolve, 0));

        expect(result.current.catalog?.pipelines).toHaveLength(1);
        expect(result.current.catalog?.pipelines[0].id).toBe('face_analysis');
    });
});

describe('Pipeline Context Integration', () => {
    it('should provide utility functions', () => {
        const { result } = renderHook(() => usePipelineContext(), {
            wrapper: createWrapper(),
        });

        expect(result.current.isPipelineAvailable).toBeDefined();
        expect(result.current.getPipeline).toBeDefined();
        expect(result.current.getFeatureFlag).toBeDefined();
        expect(result.current.refreshCatalog).toBeDefined();
        expect(result.current.clearAllCache).toBeDefined();
    });

    it('should handle pipeline availability checks', () => {
        const { result } = renderHook(() => usePipelineContext(), {
            wrapper: createWrapper(),
        });

        // Test with no data loaded
        expect(result.current.isPipelineAvailable('nonexistent')).toBe(false);
    });
});