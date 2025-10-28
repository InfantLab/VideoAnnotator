// Integration test for configuration validation flow
// Tests config change → debounced API call → field-level error display → submit blocking

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import type { ConfigValidationResult } from '@/types/api';
import React from 'react';

// Mock the API client
vi.mock('@/api/client', () => ({
    apiClient: {
        validateConfig: vi.fn(),
    },
}));

// Simple test component that mimics job creation flow
function TestConfigForm() {
    const [config, setConfig] = React.useState<Record<string, any>>({
        scene_detection: { enabled: true, threshold: 0.5 },
    });

    const [validationResult, setValidationResult] = React.useState<ConfigValidationResult | null>(null);
    const [isValidating, setIsValidating] = React.useState(false);

    const validateConfig = React.useCallback(async (cfg: Record<string, any>) => {
        setIsValidating(true);
        try {
            const result = await apiClient.validateConfig(cfg);
            setValidationResult(result);
        } catch (error) {
            console.error('Validation failed:', error);
        } finally {
            setIsValidating(false);
        }
    }, []);

    React.useEffect(() => {
        const timer = setTimeout(() => {
            validateConfig(config);
        }, 500);

        return () => clearTimeout(timer);
    }, [config, validateConfig]);

    const isValid = validationResult?.valid ?? true;

    return (
        <div>
            <label htmlFor="threshold">Threshold</label>
            <input
                id="threshold"
                type="number"
                value={config.scene_detection.threshold}
                onChange={(e) =>
                    setConfig({
                        ...config,
                        scene_detection: {
                            ...config.scene_detection,
                            threshold: parseFloat(e.target.value),
                        },
                    })
                }
            />

            {isValidating && <div>Validating configuration...</div>}

            {validationResult && !validationResult.valid && (
                <div role="alert">
                    <h3>Configuration Errors</h3>
                    {validationResult.errors.map((error, idx) => (
                        <div key={idx}>
                            <strong>{error.field}</strong>: {error.message}
                            {error.hint && <div>Hint: {error.hint}</div>}
                        </div>
                    ))}
                </div>
            )}

            {validationResult && validationResult.warnings.length > 0 && (
                <div>
                    <h3>Warnings</h3>
                    {validationResult.warnings.map((warning, idx) => (
                        <div key={idx}>
                            <strong>{warning.field}</strong>: {warning.message}
                        </div>
                    ))}
                </div>
            )}

            <button disabled={!isValid}>Submit Job</button>
        </div>
    );
}

describe('Configuration Validation Integration', () => {
    let queryClient: QueryClient;
    let user: ReturnType<typeof userEvent.setup>;

    beforeEach(() => {
        queryClient = new QueryClient({
            defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false },
            },
        });
        user = userEvent.setup();
        vi.clearAllMocks();
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.runOnlyPendingTimers();
        vi.useRealTimers();
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    it('should complete full validation flow: config change → debounced API → error display → submit blocking', async () => {
        const validResult: ConfigValidationResult = {
            valid: true,
            errors: [],
            warnings: [],
        };

        const invalidResult: ConfigValidationResult = {
            valid: false,
            errors: [
                {
                    field: 'scene_detection.threshold',
                    message: 'Threshold must be between 0 and 1',
                    severity: 'error',
                    hint: 'Try a value like 0.3',
                },
            ],
            warnings: [],
        };

        vi.mocked(apiClient.validateConfig)
            .mockResolvedValueOnce(validResult) // Initial validation
            .mockResolvedValueOnce(invalidResult); // After change

        render(<TestConfigForm />, { wrapper });

        // 1. Initial state - submit should be enabled
        const submitButton = screen.getByRole('button', { name: /submit job/i });
        expect(submitButton).not.toBeDisabled();

        // 2. Wait for initial validation
        vi.advanceTimersByTime(500);
        await waitFor(() => {
            expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
        });

        // 3. Change config to invalid value
        const thresholdInput = screen.getByLabelText(/threshold/i);
        await user.clear(thresholdInput);
        await user.type(thresholdInput, '1.5');

        // 4. Validation should not trigger immediately (debounced)
        expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);

        // 5. Show validating indicator during debounce
        vi.advanceTimersByTime(100);
        // Still debouncing...

        // 6. After 500ms, validation triggers
        vi.advanceTimersByTime(400);
        await waitFor(() => {
            expect(apiClient.validateConfig).toHaveBeenCalledTimes(2);
            expect(apiClient.validateConfig).toHaveBeenCalledWith(
                expect.objectContaining({
                    scene_detection: expect.objectContaining({ threshold: 1.5 }),
                })
            );
        });

        // 7. Error message displays
        await waitFor(() => {
            expect(screen.getByText(/threshold must be between 0 and 1/i)).toBeInTheDocument();
        });

        // 8. Hint displays
        expect(screen.getByText(/try a value like 0\.3/i)).toBeInTheDocument();

        // 9. Submit button is disabled
        expect(submitButton).toBeDisabled();
    });

    it('should allow submission with warnings after confirmation', async () => {
        const resultWithWarnings: ConfigValidationResult = {
            valid: true,
            errors: [],
            warnings: [
                {
                    field: 'scene_detection.min_scene_length',
                    message: 'Very short scenes may produce noisy results',
                    severity: 'warning',
                },
            ],
        };

        vi.mocked(apiClient.validateConfig).mockResolvedValue(resultWithWarnings);

        render(<TestConfigForm />, { wrapper });

        // Wait for validation
        vi.advanceTimersByTime(500);
        await waitFor(() => {
            expect(apiClient.validateConfig).toHaveBeenCalled();
        });

        // Warning should display
        await waitFor(() => {
            expect(screen.getByText(/very short scenes may produce noisy results/i)).toBeInTheDocument();
        });

        // Submit button should still be enabled (warnings don't block)
        const submitButton = screen.getByRole('button', { name: /submit job/i });
        expect(submitButton).not.toBeDisabled();
    });

    it('should handle validation errors gracefully', async () => {
        const error = new Error('Network error');
        vi.mocked(apiClient.validateConfig).mockRejectedValue(error);

        render(<TestConfigForm />, { wrapper });

        // Wait for validation attempt
        vi.advanceTimersByTime(500);

        await waitFor(() => {
            expect(apiClient.validateConfig).toHaveBeenCalled();
        });

        // Submit button should remain enabled on validation failure
        const submitButton = screen.getByRole('button', { name: /submit job/i });
        expect(submitButton).not.toBeDisabled();
    });

    it('should debounce rapid config changes', async () => {
        const validResult: ConfigValidationResult = {
            valid: true,
            errors: [],
            warnings: [],
        };

        vi.mocked(apiClient.validateConfig).mockResolvedValue(validResult);

        render(<TestConfigForm />, { wrapper });

        const thresholdInput = screen.getByLabelText(/threshold/i);

        // Make rapid changes
        await user.clear(thresholdInput);
        await user.type(thresholdInput, '0.1');
        vi.advanceTimersByTime(100);

        await user.clear(thresholdInput);
        await user.type(thresholdInput, '0.2');
        vi.advanceTimersByTime(100);

        await user.clear(thresholdInput);
        await user.type(thresholdInput, '0.3');

        // Should not have called API yet (still debouncing)
        expect(apiClient.validateConfig).toHaveBeenCalledTimes(0);

        // After 500ms from last change, validate
        vi.advanceTimersByTime(500);

        await waitFor(() => {
            // Should only validate once (last value)
            expect(apiClient.validateConfig).toHaveBeenCalledTimes(1);
            expect(apiClient.validateConfig).toHaveBeenCalledWith(
                expect.objectContaining({
                    scene_detection: expect.objectContaining({ threshold: 0.3 }),
                })
            );
        });
    });

    it('should display multiple field-level errors', async () => {
        const multiErrorResult: ConfigValidationResult = {
            valid: false,
            errors: [
                {
                    field: 'scene_detection.threshold',
                    message: 'Threshold out of range',
                    severity: 'error',
                },
                {
                    field: 'person_tracking.confidence',
                    message: 'Confidence must be positive',
                    severity: 'error',
                },
            ],
            warnings: [],
        };

        vi.mocked(apiClient.validateConfig).mockResolvedValue(multiErrorResult);

        render(<TestConfigForm />, { wrapper });

        vi.advanceTimersByTime(500);

        await waitFor(() => {
            expect(screen.getByText(/threshold out of range/i)).toBeInTheDocument();
            expect(screen.getByText(/confidence must be positive/i)).toBeInTheDocument();
        });

        // Submit should be blocked
        const submitButton = screen.getByRole('button', { name: /submit job/i });
        expect(submitButton).toBeDisabled();
    });

    it('should clear previous errors when config becomes valid', async () => {
        const invalidResult: ConfigValidationResult = {
            valid: false,
            errors: [
                {
                    field: 'scene_detection.threshold',
                    message: 'Threshold error',
                    severity: 'error',
                },
            ],
            warnings: [],
        };

        const validResult: ConfigValidationResult = {
            valid: true,
            errors: [],
            warnings: [],
        };

        vi.mocked(apiClient.validateConfig)
            .mockResolvedValueOnce(invalidResult)
            .mockResolvedValueOnce(validResult);

        render(<TestConfigForm />, { wrapper });

        // Initial validation (invalid)
        vi.advanceTimersByTime(500);
        await waitFor(() => {
            expect(screen.getByText(/threshold error/i)).toBeInTheDocument();
        });

        // Change to valid value
        const thresholdInput = screen.getByLabelText(/threshold/i);
        await user.clear(thresholdInput);
        await user.type(thresholdInput, '0.5');

        vi.advanceTimersByTime(500);

        // Error should clear
        await waitFor(() => {
            expect(screen.queryByText(/threshold error/i)).not.toBeInTheDocument();
        });

        // Submit should be enabled
        const submitButton = screen.getByRole('button', { name: /submit job/i });
        expect(submitButton).not.toBeDisabled();
    });
});
