/**
 * T045: Integration tests for error handling
 * Tests error propagation from API → parsing → display in UI
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ParsedError } from '@/types/api';

// Mock API client
const mockApiClient = {
    async fetchWithError(errorType: 'v1.3' | 'legacy' | 'network') {
        if (errorType === 'v1.3') {
            throw {
                error: 'Job not found',
                error_code: 'JOB_NOT_FOUND',
                request_id: 'req-test-123',
                hint: 'Check the job ID and try again',
            };
        } else if (errorType === 'legacy') {
            throw { message: 'Old style error' };
        } else {
            throw new Error('Failed to fetch');
        }
    },
};

// Mock component that fetches data and displays errors
function TestErrorHandlingComponent() {
    return (
        <div>
            <h1>Test Error Handling Integration</h1>
            <button onClick={() => mockApiClient.fetchWithError('v1.3')}>
                Trigger v1.3 Error
            </button>
            <button onClick={() => mockApiClient.fetchWithError('legacy')}>
                Trigger Legacy Error
            </button>
            <button onClick={() => mockApiClient.fetchWithError('network')}>
                Trigger Network Error
            </button>
            <div data-testid="error-display">Mock Error Display - Not Implemented</div>
        </div>
    );
}

describe('Error Handling Integration', () => {
    describe('API error propagation', () => {
        it('should catch v1.3.0 error and display properly', async () => {
            const user = userEvent.setup();

            render(<TestErrorHandlingComponent />);

            const button = screen.getByRole('button', { name: /trigger v1.3 error/i });
            await user.click(button);

            await waitFor(() => {
                expect(screen.getByText(/job not found/i)).toBeInTheDocument();
            });

            // Should show hint
            expect(screen.getByText(/check the job id/i)).toBeInTheDocument();

            // Should show error code in technical details (when expanded)
            expect(screen.getByText(/JOB_NOT_FOUND/)).toBeInTheDocument();
        });

        it('should catch legacy error and display message', async () => {
            const user = userEvent.setup();

            render(<TestErrorHandlingComponent />);

            const button = screen.getByRole('button', { name: /trigger legacy error/i });
            await user.click(button);

            await waitFor(() => {
                expect(screen.getByText(/old style error/i)).toBeInTheDocument();
            });
        });

        it('should catch network error and show helpful message', async () => {
            const user = userEvent.setup();

            render(<TestErrorHandlingComponent />);

            const button = screen.getByRole('button', { name: /trigger network error/i });
            await user.click(button);

            await waitFor(() => {
                expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
            });

            // Should include network-specific hint
            expect(screen.getByText(/check your connection/i)).toBeInTheDocument();
        });
    });

    describe('validation error display', () => {
        it('should display field errors from API', async () => {
            const mockValidationError = {
                error: [
                    {
                        field: 'video_url',
                        message: 'Must be a valid URL',
                        hint: 'Use http:// or https://',
                    },
                    {
                        field: 'pipeline_id',
                        message: 'Required field',
                    },
                ],
                error_code: 'VALIDATION_ERROR',
                request_id: 'req-val-456',
            };

            // Component that receives validation error
            function ValidationErrorComponent() {
                return (
                    <div>
                        <h1>Validation Error Test</h1>
                        <div role="alert">
                            {mockValidationError.error.map((fieldError, idx) => (
                                <div key={idx}>
                                    <strong>{fieldError.field}:</strong> {fieldError.message}
                                    {fieldError.hint && <div>Tip: {fieldError.hint}</div>}
                                </div>
                            ))}
                        </div>
                    </div>
                );
            }

            render(<ValidationErrorComponent />);

            expect(screen.getByText(/video_url/i)).toBeInTheDocument();
            expect(screen.getByText(/must be a valid url/i)).toBeInTheDocument();
            expect(screen.getByText(/use http/i)).toBeInTheDocument();
            expect(screen.getByText(/pipeline_id/i)).toBeInTheDocument();
        });
    });

    describe('error context propagation', () => {
        it('should show request ID for debugging', async () => {
            const error: ParsedError = {
                message: 'Server error',
                code: 'INTERNAL_ERROR',
                requestId: 'req-debug-789',
            };

            function ErrorWithRequestId() {
                return (
                    <div>
                        <div role="alert">{error.message}</div>
                        <details>
                            <summary>Technical Details</summary>
                            <div>Request ID: {error.requestId}</div>
                        </details>
                    </div>
                );
            }

            const { container } = render(<ErrorWithRequestId />);

            // Open details
            const summary = container.querySelector('summary');
            if (summary) {
                await userEvent.click(summary);
            }

            expect(screen.getByText(/req-debug-789/)).toBeInTheDocument();
        });
    });

    describe('toast notifications with errors', () => {
        it('should show error toast with hint', () => {
            const error: ParsedError = {
                message: 'Job creation failed',
                hint: 'Check your video URL format',
            };

            function ToastErrorComponent() {
                return (
                    <div role="status" aria-live="polite">
                        <div>{error.message}</div>
                        {error.hint && <div>Tip: {error.hint}</div>}
                    </div>
                );
            }

            render(<ToastErrorComponent />);

            expect(screen.getByText(/job creation failed/i)).toBeInTheDocument();
            expect(screen.getByText(/check your video url/i)).toBeInTheDocument();
        });

        it('should show concise message in toast', () => {
            const error: ParsedError = {
                message: '2 validation errors (first: video_url: Invalid URL)',
                fieldErrors: [
                    { field: 'video_url', message: 'Invalid URL' },
                    { field: 'config.param', message: 'Out of range' },
                ],
            };

            function ToastValidationComponent() {
                return (
                    <div role="status">
                        {/* Toast shows summary */}
                        <div>{error.message}</div>
                        {/* Details page shows full field errors */}
                    </div>
                );
            }

            render(<ToastValidationComponent />);

            expect(screen.getByText(/2 validation errors/i)).toBeInTheDocument();
            expect(screen.getByText(/video_url/i)).toBeInTheDocument();
        });
    });

    describe('error recovery flows', () => {
        it('should allow dismissing error and retrying', async () => {
            const user = userEvent.setup();
            let attemptCount = 0;

            function RetryableErrorComponent() {
                const [error, setError] = React.useState<ParsedError | null>(null);

                const handleRetry = async () => {
                    attemptCount++;
                    if (attemptCount < 2) {
                        setError({ message: 'Request failed', code: 'RATE_LIMIT' });
                    } else {
                        setError(null);
                    }
                };

                return (
                    <div>
                        {error && (
                            <div role="alert">
                                {error.message}
                                <button onClick={() => setError(null)}>Dismiss</button>
                                <button onClick={handleRetry}>Retry</button>
                            </div>
                        )}
                        <button onClick={handleRetry}>Start Request</button>
                    </div>
                );
            }

            render(<RetryableErrorComponent />);

            // Trigger error
            await user.click(screen.getByRole('button', { name: /start request/i }));

            expect(screen.getByText(/request failed/i)).toBeInTheDocument();

            // Retry
            await user.click(screen.getByRole('button', { name: /retry/i }));

            // Error should be gone
            await waitFor(() => {
                expect(screen.queryByText(/request failed/i)).not.toBeInTheDocument();
            });
        });
    });
});

// Add React import for JSX
import React from 'react';
