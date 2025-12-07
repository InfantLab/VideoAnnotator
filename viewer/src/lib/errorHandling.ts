// Error handling utilities for VideoAnnotator v1.3.0
// Defensive parsing of API errors with backward compatibility

import { z } from 'zod';
import { ErrorEnvelopeSchema } from './validation';
import type {
  ErrorEnvelope,
  ParsedError,
  FieldError,
} from '@/types/api';

/**
 * Parse API error response defensively
 * Supports both v1.3.0 ErrorEnvelope and legacy formats
 * 
 * Handles:
 * - v1.3.0 ErrorEnvelope: { error: string | FieldError[], error_code?, request_id?, hint? }
 * - Legacy object: { error: string } or { message: string }
 * - Plain string: "Error message"
 * - Unknown/malformed responses
 * 
 * @param error - API error response (unknown type)
 * @returns ParsedError with consistent structure
 */
export function parseApiError(error: unknown): ParsedError {
  // Handle null/undefined
  if (error === null || error === undefined) {
    return {
      message: 'An unknown error occurred',
    };
  }

  // Handle string errors
  if (typeof error === 'string') {
    return {
      message: error,
    };
  }

  // Handle Error objects
  if (error instanceof Error) {
    return {
      message: error.message,
    };
  }

  // Handle objects
  if (typeof error === 'object') {
    // Try v1.3.0 ErrorEnvelope format with Zod (defensive)
    const envelopeResult = ErrorEnvelopeSchema.safeParse(error);
    if (envelopeResult.success) {
      const env = envelopeResult.data;
      // Extract primary message
      let message = 'An error occurred';
      if (typeof env.error === 'string') {
        message = env.error;
      } else if (Array.isArray(env.error) && env.error.length > 0) {
        message = env.error[0].message;
      }

      return {
        message,
        code: env.error_code,
        requestId: env.request_id,
        hint: env.hint,
        details: env.error // Keep full details
      };
    }

    // Handle legacy/other object formats
    const errObj = error as any;
    const message = errObj.message || errObj.error || errObj.detail || JSON.stringify(error);
    
    return {
      message: String(message),
      code: errObj.code || errObj.status || errObj.statusCode,
    };
  }

  return {
    message: String(error),
  };
}

/**
 * Parse validated ErrorEnvelope into ParsedError
 */
function parseErrorEnvelope(envelope: ErrorEnvelope): ParsedError {
  const { error, error_code, request_id, hint } = envelope;

  // Handle field errors (array)
  if (Array.isArray(error)) {
    return {
      message: buildFieldErrorMessage(error),
      code: error_code,
      requestId: request_id,
      hint,
      fieldErrors: error,
    };
  }

  // Handle string error
  return {
    message: error,
    code: error_code,
    requestId: request_id,
    hint,
  };
}

/**
 * Build human-readable message from field errors
 */
function buildFieldErrorMessage(fieldErrors: FieldError[]): string {
  if (fieldErrors.length === 0) {
    return 'Validation failed';
  }

  if (fieldErrors.length === 1) {
    const { field, message } = fieldErrors[0];
    return `${field}: ${message}`;
  }

  // Multiple errors: list them
  const errorCount = fieldErrors.length;
  const firstError = fieldErrors[0];
  return `${errorCount} validation errors (first: ${firstError.field}: ${firstError.message})`;
}

/**
 * Format error for user display
 * @param error - ParsedError
 * @returns Formatted string with message and optional hint
 */
export function formatErrorForDisplay(error: ParsedError): string {
  let message = error.message;

  if (error.hint) {
    message += `\n\nTip: ${error.hint}`;
  }

  return message;
}

/**
 * Format error for logging/debugging
 * Includes all technical details
 */
export function formatErrorForLogging(error: ParsedError): string {
  const parts: string[] = [error.message];

  if (error.code) {
    parts.push(`Code: ${error.code}`);
  }

  if (error.requestId) {
    parts.push(`Request ID: ${error.requestId}`);
  }

  if (error.hint) {
    parts.push(`Hint: ${error.hint}`);
  }

  if (error.fieldErrors && error.fieldErrors.length > 0) {
    parts.push('\nField Errors:');
    error.fieldErrors.forEach(({ field, message, hint }) => {
      parts.push(`  - ${field}: ${message}`);
      if (hint) {
        parts.push(`    Hint: ${hint}`);
      }
    });
  }

  return parts.join('\n');
}

/**
 * Check if error is a network error
 */
export function isNetworkError(error: unknown): boolean {
  if (error instanceof Error) {
    return (
      error.message.includes('network') ||
      error.message.includes('fetch') ||
      error.message.includes('NetworkError') ||
      error.message.includes('Failed to fetch')
    );
  }
  return false;
}

/**
 * Check if error is an authentication error
 */
export function isAuthError(error: ParsedError): boolean {
  if (error.code) {
    return (
      error.code === 'AUTH_FAILED' ||
      error.code === 'UNAUTHORIZED' ||
      error.code === 'TOKEN_INVALID' ||
      error.code === 'TOKEN_EXPIRED'
    );
  }

  const messageLower = error.message.toLowerCase();
  return (
    messageLower.includes('unauthorized') ||
    messageLower.includes('authentication') ||
    messageLower.includes('token') ||
    messageLower.includes('401')
  );
}

/**
 * Check if error is a validation error
 */
export function isValidationError(error: ParsedError): boolean {
  return (
    error.fieldErrors !== undefined &&
    error.fieldErrors.length > 0
  );
}

/**
 * Extract retry-able status from error
 * Returns true if the operation should be retried
 */
export function isRetryableError(error: ParsedError): boolean {
  if (error.code) {
    const retryableCodes = [
      'RATE_LIMIT',
      'SERVER_BUSY',
      'TIMEOUT',
      'SERVICE_UNAVAILABLE',
    ];
    return retryableCodes.includes(error.code);
  }

  const messageLower = error.message.toLowerCase();
  return (
    messageLower.includes('timeout') ||
    messageLower.includes('rate limit') ||
    messageLower.includes('too many requests') ||
    messageLower.includes('503') ||
    messageLower.includes('429')
  );
}
