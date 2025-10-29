/**
 * T051: Toast notification helpers for displaying errors with hints
 * Provides consistent error toast display across the application
 */

import type { ParsedError } from '@/types/api';

export interface ToastFunction {
  (props: {
    title?: string;
    description?: string;
    variant?: 'default' | 'destructive';
  }): void;
}

/**
 * Show an error toast with proper formatting for ParsedError
 * Includes hint as part of description if available
 */
export function showErrorToast(toast: ToastFunction, error: ParsedError) {
  const description = error.hint 
    ? `${error.message}\n\n💡 Tip: ${error.hint}`
    : error.message;

  toast({
    title: 'Error',
    description,
    variant: 'destructive',
  });
}

/**
 * Show a success toast
 */
export function showSuccessToast(toast: ToastFunction, message: string, description?: string) {
  toast({
    title: message,
    description,
    variant: 'default',
  });
}

/**
 * Show a validation error toast with field-level details
 */
export function showValidationErrorToast(toast: ToastFunction, error: ParsedError) {
  if (error.fieldErrors && error.fieldErrors.length > 0) {
    const fieldCount = error.fieldErrors.length;
    const firstError = error.fieldErrors[0];
    
    const description = fieldCount === 1
      ? `${firstError.field}: ${firstError.message}${firstError.hint ? `\n💡 ${firstError.hint}` : ''}`
      : `${fieldCount} validation errors. First: ${firstError.field}: ${firstError.message}`;

    toast({
      title: 'Validation Error',
      description,
      variant: 'destructive',
    });
  } else {
    showErrorToast(toast, error);
  }
}
