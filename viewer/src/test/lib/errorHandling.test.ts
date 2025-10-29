/**
 * T043: Unit tests for error handling utilities
 * Testing parseApiError, formatters, and error type checks
 */

import { describe, it, expect } from 'vitest';
import {
    parseApiError,
    formatErrorForDisplay,
    formatErrorForLogging,
    isNetworkError,
    isAuthError,
    isValidationError,
    isRetryableError,
} from '@/lib/errorHandling';
import type { ParsedError, FieldError } from '@/types/api';

describe('parseApiError', () => {
    describe('v1.3.0 ErrorEnvelope format', () => {
        it('should parse basic error with message', () => {
            const apiError = {
                error: 'Job not found',
                error_code: 'JOB_NOT_FOUND',
                request_id: 'req-12345',
            };

            const parsed = parseApiError(apiError);

            expect(parsed.message).toBe('Job not found');
            expect(parsed.code).toBe('JOB_NOT_FOUND');
            expect(parsed.requestId).toBe('req-12345');
            expect(parsed.hint).toBeUndefined();
            expect(parsed.fieldErrors).toBeUndefined();
        });

        it('should parse error with hint', () => {
            const apiError = {
                error: 'Invalid pipeline configuration',
                error_code: 'CONFIG_INVALID',
                hint: 'Check that all required parameters are provided',
            };

            const parsed = parseApiError(apiError);

            expect(parsed.message).toBe('Invalid pipeline configuration');
            expect(parsed.hint).toBe('Check that all required parameters are provided');
        });

        it('should parse field errors (validation)', () => {
            const fieldErrors: FieldError[] = [
                {
                    field: 'video_url',
                    message: 'Must be a valid URL',
                    hint: 'Use http:// or https:// scheme',
                },
                {
                    field: 'pipeline_id',
                    message: 'Pipeline not found',
                },
            ];

            const apiError = {
                error: fieldErrors,
                error_code: 'VALIDATION_ERROR',
                request_id: 'req-67890',
            };

            const parsed = parseApiError(apiError);

            expect(parsed.fieldErrors).toEqual(fieldErrors);
            expect(parsed.message).toContain('2 validation errors');
            expect(parsed.message).toContain('video_url');
            expect(parsed.code).toBe('VALIDATION_ERROR');
        });

        it('should handle single field error', () => {
            const apiError = {
                error: [
                    {
                        field: 'token',
                        message: 'Token is required',
                    },
                ],
                error_code: 'VALIDATION_ERROR',
            };

            const parsed = parseApiError(apiError);

            expect(parsed.message).toBe('token: Token is required');
            expect(parsed.fieldErrors).toHaveLength(1);
        });
    });

    describe('legacy error formats', () => {
        it('should parse {error: string} format', () => {
            const apiError = { error: 'Server error occurred' };

            const parsed = parseApiError(apiError);

            expect(parsed.message).toBe('Server error occurred');
            expect(parsed.code).toBeUndefined();
        });

        it('should parse {message: string} format', () => {
            const apiError = { message: 'Bad request' };

            const parsed = parseApiError(apiError);

            expect(parsed.message).toBe('Bad request');
        });

        it('should parse {detail: string} format', () => {
            const apiError = { detail: 'Database connection failed' };

            const parsed = parseApiError(apiError);

            expect(parsed.message).toBe('Database connection failed');
        });

        it('should parse plain string', () => {
            const parsed = parseApiError('Something went wrong');

            expect(parsed.message).toBe('Something went wrong');
        });

        it('should parse Error object', () => {
            const error = new Error('Network timeout');

            const parsed = parseApiError(error);

            expect(parsed.message).toBe('Network timeout');
        });
    });

    describe('edge cases', () => {
        it('should handle null', () => {
            const parsed = parseApiError(null);

            expect(parsed.message).toBe('An unknown error occurred');
        });

        it('should handle undefined', () => {
            const parsed = parseApiError(undefined);

            expect(parsed.message).toBe('An unknown error occurred');
        });

        it('should handle empty object', () => {
            const parsed = parseApiError({});

            // Empty object gets stringified as "{}"
            expect(parsed.message).toBe('{}');
        });

        it('should handle malformed ErrorEnvelope', () => {
            const apiError = {
                error: null,
                error_code: 'SOME_CODE',
            };

            const parsed = parseApiError(apiError);

            expect(parsed).toBeDefined();
            // With error: null, it doesn't pass Zod validation so code is not extracted
            // Instead it gets stringified
            expect(parsed.message).toBeDefined();
        });
    });
});

describe('formatErrorForDisplay', () => {
    it('should format message only', () => {
        const error: ParsedError = {
            message: 'Job failed',
        };

        const formatted = formatErrorForDisplay(error);

        expect(formatted).toBe('Job failed');
    });

    it('should include hint when present', () => {
        const error: ParsedError = {
            message: 'Pipeline not found',
            hint: 'Use /api/v1/pipelines to see available pipelines',
        };

        const formatted = formatErrorForDisplay(error);

        expect(formatted).toContain('Pipeline not found');
        expect(formatted).toContain('Tip: Use /api/v1/pipelines');
    });
});

describe('formatErrorForLogging', () => {
    it('should format basic error', () => {
        const error: ParsedError = {
            message: 'Job failed',
            code: 'JOB_FAILED',
            requestId: 'req-abc',
        };

        const formatted = formatErrorForLogging(error);

        expect(formatted).toContain('Job failed');
        expect(formatted).toContain('Code: JOB_FAILED');
        expect(formatted).toContain('Request ID: req-abc');
    });

    it('should format field errors', () => {
        const error: ParsedError = {
            message: '2 validation errors',
            code: 'VALIDATION_ERROR',
            fieldErrors: [
                { field: 'video_url', message: 'Invalid URL' },
                { field: 'pipeline_id', message: 'Required', hint: 'Choose a pipeline' },
            ],
        };

        const formatted = formatErrorForLogging(error);

        expect(formatted).toContain('Field Errors:');
        expect(formatted).toContain('video_url: Invalid URL');
        expect(formatted).toContain('pipeline_id: Required');
        expect(formatted).toContain('Hint: Choose a pipeline');
    });
});

describe('isNetworkError', () => {
    it('should detect network errors', () => {
        expect(isNetworkError(new Error('Failed to fetch'))).toBe(true);
        expect(isNetworkError(new Error('NetworkError: Connection refused'))).toBe(true);
        expect(isNetworkError(new Error('network timeout'))).toBe(true);
    });

    it('should return false for non-network errors', () => {
        expect(isNetworkError(new Error('Validation failed'))).toBe(false);
        expect(isNetworkError('some string')).toBe(false);
        expect(isNetworkError(null)).toBe(false);
    });
});

describe('isAuthError', () => {
    it('should detect auth errors by code', () => {
        const error: ParsedError = {
            message: 'Unauthorized',
            code: 'AUTH_FAILED',
        };

        expect(isAuthError(error)).toBe(true);
    });

    it('should detect auth errors by message', () => {
        const error: ParsedError = {
            message: '401 Unauthorized',
        };

        expect(isAuthError(error)).toBe(true);
    });

    it('should return false for non-auth errors', () => {
        const error: ParsedError = {
            message: 'Job not found',
        };

        expect(isAuthError(error)).toBe(false);
    });
});

describe('isValidationError', () => {
    it('should detect validation errors', () => {
        const error: ParsedError = {
            message: 'Validation failed',
            fieldErrors: [{ field: 'test', message: 'Required' }],
        };

        expect(isValidationError(error)).toBe(true);
    });

    it('should return false for non-validation errors', () => {
        const error: ParsedError = {
            message: 'Server error',
        };

        expect(isValidationError(error)).toBe(false);
    });
});

describe('isRetryableError', () => {
    it('should detect retryable errors by code', () => {
        const error: ParsedError = {
            message: 'Too busy',
            code: 'RATE_LIMIT',
        };

        expect(isRetryableError(error)).toBe(true);
    });

    it('should detect retryable errors by message', () => {
        const error: ParsedError = {
            message: '503 Service unavailable',
        };

        expect(isRetryableError(error)).toBe(true);
    });

    it('should return false for non-retryable errors', () => {
        const error: ParsedError = {
            message: 'Job not found',
            code: 'JOB_NOT_FOUND',
        };

        expect(isRetryableError(error)).toBe(false);
    });
});
