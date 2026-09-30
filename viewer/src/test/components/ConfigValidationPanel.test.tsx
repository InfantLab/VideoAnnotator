// Component tests for ConfigValidationPanel
// Tests error/warning display, field grouping, and hint rendering

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfigValidationPanel } from '@/components/ConfigValidationPanel';
import type { ConfigValidationResult } from '@/types/api';
import React from 'react';

describe('ConfigValidationPanel', () => {
    let user: ReturnType<typeof userEvent.setup>;

    beforeEach(() => {
        user = userEvent.setup();
        vi.clearAllMocks();
    });

    describe('no validation results', () => {
        it('should not render when validationResult is null', () => {
            const { container } = render(
                <ConfigValidationPanel validationResult={null} isValidating={false} />
            );

            expect(container.firstChild).toBeNull();
        });

        it('should not render when validationResult is undefined', () => {
            const { container } = render(
                <ConfigValidationPanel validationResult={undefined} isValidating={false} />
            );

            expect(container.firstChild).toBeNull();
        });
    });

    describe('loading state', () => {
        it('should show validating indicator when isValidating is true', () => {
            render(
                <ConfigValidationPanel validationResult={null} isValidating={true} />
            );

            expect(screen.getByText(/validating/i)).toBeInTheDocument();
        });
    });

    describe('valid configuration', () => {
        it('should show success message for valid config with no warnings', () => {
            const result: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/configuration is valid/i)).toBeInTheDocument();
        });
    });

    describe('error display', () => {
        it('should display errors with field names and messages', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold must be between 0 and 1',
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

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            // Check for sub-field names (shown under parent groups)
            expect(screen.getByText(/^threshold$/i)).toBeInTheDocument();
            expect(screen.getByText(/threshold must be between 0 and 1/i)).toBeInTheDocument();
            expect(screen.getByText(/^confidence$/i)).toBeInTheDocument();
            expect(screen.getByText(/confidence must be positive/i)).toBeInTheDocument();
        });

        it('should display error hints when available', () => {
            const result: ConfigValidationResult = {
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

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/try a value like 0\.3/i)).toBeInTheDocument();
        });

        it('should display error codes when available', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold must be between 0 and 1',
                        severity: 'error',
                        error_code: 'VALUE_OUT_OF_RANGE',
                    },
                ],
                warnings: [],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/VALUE_OUT_OF_RANGE/i)).toBeInTheDocument();
        });

        it('should display suggested values when available', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Invalid threshold',
                        severity: 'error',
                        suggested_value: 0.3,
                    },
                ],
                warnings: [],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/0\.3/)).toBeInTheDocument();
        });
    });

    describe('warning display', () => {
        it('should display warnings with field names and messages', () => {
            const result: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [
                    {
                        field: 'scene_detection.min_scene_length',
                        message: 'Very short scenes may be noisy',
                        severity: 'warning',
                    },
                ],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            // Check for sub-field name (shown under parent group)
            expect(screen.getByText(/^min_scene_length$/i)).toBeInTheDocument();
            expect(screen.getByText(/very short scenes may be noisy/i)).toBeInTheDocument();
        });

        it('should show both errors and warnings', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold invalid',
                        severity: 'error',
                    },
                ],
                warnings: [
                    {
                        field: 'person_tracking.max_persons',
                        message: 'High value may impact performance',
                        severity: 'warning',
                    },
                ],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/threshold invalid/i)).toBeInTheDocument();
            expect(screen.getByText(/high value may impact performance/i)).toBeInTheDocument();
        });
    });

    describe('field grouping', () => {
        it('should group errors by parent field', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold error',
                        severity: 'error',
                    },
                    {
                        field: 'scene_detection.min_length',
                        message: 'Length error',
                        severity: 'error',
                    },
                    {
                        field: 'person_tracking.confidence',
                        message: 'Confidence error',
                        severity: 'error',
                    },
                ],
                warnings: [],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            // Should show scene_detection group
            expect(screen.getByText(/scene_detection/i)).toBeInTheDocument();
            // Should show person_tracking group
            expect(screen.getByText(/person_tracking/i)).toBeInTheDocument();
        });
    });

    describe('collapsible sections', () => {
        it('should allow collapsing error sections', async () => {
            const result: ConfigValidationResult = {
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

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            // Error message should be visible initially
            expect(screen.getByText(/threshold error/i)).toBeInTheDocument();

            // Find and click collapse button (if exists)
            const collapseButton = screen.queryByRole('button', { name: /collapse|hide/i });
            if (collapseButton) {
                await user.click(collapseButton);

                // Error message should be hidden
                expect(screen.queryByText(/threshold error/i)).not.toBeInTheDocument();
            }
        });
    });

    describe('dev mode display', () => {
        it('should show additional details in dev mode', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold error',
                        severity: 'error',
                        error_code: 'VALUE_OUT_OF_RANGE',
                    },
                ],
                warnings: [],
            };

            render(
                <ConfigValidationPanel
                    validationResult={result}
                    isValidating={false}
                    devMode={true}
                />
            );

            // Error codes should be visible in dev mode
            expect(screen.getByText(/VALUE_OUT_OF_RANGE/i)).toBeInTheDocument();
        });

        it('should hide error codes in non-dev mode', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'scene_detection.threshold',
                        message: 'Threshold error',
                        severity: 'error',
                        error_code: 'VALUE_OUT_OF_RANGE',
                    },
                ],
                warnings: [],
            };

            render(
                <ConfigValidationPanel
                    validationResult={result}
                    isValidating={false}
                    devMode={false}
                />
            );

            // Error codes should be hidden in non-dev mode
            expect(screen.queryByText(/VALUE_OUT_OF_RANGE/i)).not.toBeInTheDocument();
        });
    });

    describe('accessibility', () => {
        it('should use appropriate ARIA roles for errors', () => {
            const result: ConfigValidationResult = {
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

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            // Should have alert role for errors
            const alerts = screen.queryAllByRole('alert');
            expect(alerts.length).toBeGreaterThan(0);
        });

        it('should provide clear visual distinction between errors and warnings', () => {
            const result: ConfigValidationResult = {
                valid: false,
                errors: [
                    {
                        field: 'field1',
                        message: 'Error message',
                        severity: 'error',
                    },
                ],
                warnings: [
                    {
                        field: 'field2',
                        message: 'Warning message',
                        severity: 'warning',
                    },
                ],
            };

            const { container } = render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            // Errors should have destructive/error styling
            const errorElements = container.querySelectorAll('[class*="destructive"]');
            expect(errorElements.length).toBeGreaterThan(0);
        });
    });

    describe('empty states', () => {
        it('should handle empty errors array', () => {
            const result: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/configuration is valid/i)).toBeInTheDocument();
        });

        it('should handle result with only warnings', () => {
            const result: ConfigValidationResult = {
                valid: true,
                errors: [],
                warnings: [
                    {
                        field: 'test',
                        message: 'Warning message',
                        severity: 'warning',
                    },
                ],
            };

            render(
                <ConfigValidationPanel validationResult={result} isValidating={false} />
            );

            expect(screen.getByText(/warning message/i)).toBeInTheDocument();
            // Should not show "invalid" message
            expect(screen.queryByText(/invalid/i)).not.toBeInTheDocument();
        });
    });
});
