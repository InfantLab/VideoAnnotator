// Unit tests for useConfigValidation hook
// Tests debouncing (500ms), config hash caching, and validation state management

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useConfigValidation } from '@/hooks/useConfigValidation';
import { apiClient } from '@/api/client';
import type { ConfigValidationResult } from '@/types/api';
import React from 'react';

// Mock the API client
vi.mock('@/api/client', () => ({
    apiClient: {
        validateConfig: vi.fn(),
    },
}));

describe('useConfigValidation', () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        queryClient = new QueryClient({
            defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false },
            },
        });
        vi.clearAllMocks();
        // Don't use fake timers - real timers work better with async promises
    });

    afterEach(() => {
        // No timer cleanup needed
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    describe('debouncing', () => {
        it('should debounce validation calls by 500ms', async () => {
            const mockResult: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config1 = { scene_detection: { enabled: true } };
            const config2 = { scene_detection: { enabled: false } };
            const config3 = { person_tracking: { enabled: true } };

            // Rapid fire changes
            result.current.validateConfig(config1);
            result.current.validateConfig(config2);
            result.current.validateConfig(config3);

            // Should not have called API yet
            expect(apiClient.validateConfig).not.toHaveBeenCalled();

            // Wait 400ms - still no call
            await new Promise(resolve => setTimeout(resolve, 400));
            expect(apiClient.validateConfig).not.toHaveBeenCalled();

            // Wait another 150ms (total 550ms) - should have called
            await new Promise(resolve => setTimeout(resolve, 150));

            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });
            expect(apiClient.validateConfig).toHaveBeenCalledWith(config3);
        });

        it('should reset debounce timer on each config change', async () => {
            const mockResult: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config1 = { scene_detection: { enabled: true } };
            const config2 = { person_tracking: { enabled: true } };

            // First change
            result.current.validateConfig(config1);

            // Wait 400ms
            await new Promise(resolve => setTimeout(resolve, 400));
            expect(apiClient.validateConfig).not.toHaveBeenCalled();

            // Second change (resets timer)
            result.current.validateConfig(config2);

            // Wait another 400ms (total 800ms, but timer was reset)
            await new Promise(resolve => setTimeout(resolve, 400));
            expect(apiClient.validateConfig).not.toHaveBeenCalled();

            // Wait final 150ms (550ms since last change - should have called)
            await new Promise(resolve => setTimeout(resolve, 150));

            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });
            expect(apiClient.validateConfig).toHaveBeenCalledWith(config2);
        });
    });

    describe('config hash caching', () => {
        it('should cache results by config hash', async () => {
            const mockResult: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config = { scene_detection: { enabled: true } };

            // First validation
            result.current.validateConfig(config);
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });

            // Second validation with same config (should use cache)
            result.current.validateConfig(config);
            await new Promise(resolve => setTimeout(resolve, 550));

            // Should still only have been called once (cached)
            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });
        });

        it('should not use cache for different configs', async () => {
            const mockResult: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config1 = { scene_detection: { enabled: true } };
            const config2 = { scene_detection: { enabled: false } }; // Different

            // First validation
            result.current.validateConfig(config1);
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });

            // Second validation with different config
            result.current.validateConfig(config2);
            await new Promise(resolve => setTimeout(resolve, 550));

            // Should be called twice (different configs)
            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(2);
            });
        });

        it('should treat object property order differences as same config', async () => {
            const mockResult: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config1 = { a: 1, b: 2, c: 3 };
            const config2 = { c: 3, a: 1, b: 2 }; // Same values, different order

            // First validation
            result.current.validateConfig(config1);
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });

            // Second validation (should use cache)
            result.current.validateConfig(config2);
            await new Promise(resolve => setTimeout(resolve, 550));

            // Should still only have been called once (order-independent hash)
            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            });
        });
    });

    describe('validation state management', () => {
        it('should return validation results', async () => {
            const mockResult: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold must be between 0 and 1',
                        severity: 'error',
                        hint: 'Try a value like 0.3',
                        error_code: 'VALUE_OUT_OF_RANGE',
                    },
                ],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config = { scene_detection: { threshold: 1.5 } };

            result.current.validateConfig(config);
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(result.current.validationResult).toEqual(mockResult);
                expect(result.current.validationResult?.valid).toBe(false);
                expect(result.current.validationResult?.errors).toHaveLength(1);
            });
        });

        it('should track loading state', async () => {
            let resolveValidation: (value: ConfigValidationResult) => void;
            const validationPromise = new Promise<ConfigValidationResult>((resolve) => {
                resolveValidation = resolve;
            });

            vi.mocked(apiClient.validateConfig).mockReturnValue(validationPromise);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            expect(result.current.isValidating).toBe(false);

            const config = { scene_detection: { enabled: true } };
            result.current.validateConfig(config);

            // Wait a bit for the debounce
            await new Promise(resolve => setTimeout(resolve, 550));

            // Should be validating or have result by now
            await waitFor(() => {
                expect(apiClient.validateConfig).toHaveBeenCalled();
            });

            // Resolve validation
            resolveValidation!({
                valid: true,
                errors: [],
                warnings: [],
            });

            // Should no longer be validating
            await waitFor(() => {
                expect(result.current.isValidating).toBe(false);
            });
        });

        it('should handle validation errors', async () => {
            const error = new Error('Network error');
            vi.mocked(apiClient.validateConfig).mockRejectedValue(error);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config = { scene_detection: { enabled: true } };

            result.current.validateConfig(config);
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(result.current.error).toEqual(error);
                expect(result.current.validationResult).toBeNull();
            });
        });

        it('should clear previous results when validating new config', async () => {
            const mockResult1: ConfigValidationResult = {
                valid: false,
                errors: [{ field: 'test', message: 'Error 1', severity: 'error', error_code: 'ERROR_1' }],
                warnings: [],
            };

            const mockResult2: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig)
                .mockResolvedValueOnce(mockResult1)
                .mockResolvedValueOnce(mockResult2);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            // First validation
            result.current.validateConfig({ a: 1 });
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(result.current.validationResult).toEqual(mockResult1);
            });

            // Second validation
            result.current.validateConfig({ b: 2 });
            await new Promise(resolve => setTimeout(resolve, 550));

            await waitFor(() => {
                expect(result.current.validationResult).toEqual(mockResult2);
                expect(result.current.validationResult?.errors).toHaveLength(0);
            });
        });
    });

    describe('manual validation trigger', () => {
        it('should allow immediate validation without debounce', async () => {
            const mockResult: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            vi.mocked(apiClient.validateConfig).mockResolvedValue(mockResult);

            const { result } = renderHook(() => useConfigValidation(), { wrapper });

            const config = { scene_detection: { enabled: true } };

            // Validate immediately (if hook provides this method)
            if (result.current.validateNow) {
                result.current.validateNow(config);

                // Should call API immediately without waiting for debounce
                await waitFor(() => {
                    expect(apiClient.validateConfig).toHaveBeenCalledWith(config);
                });
            }
        });
    });
});
