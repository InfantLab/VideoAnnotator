/**
 * T051: Toast notification helpers for displaying errors with hints
 * Provides consistent error toast display across the application
 */

import React from 'react';
import type { ParsedError } from '@/types/api';
import { toast as toastFn } from '@/hooks/use-toast';

// Type for toast function - uses ReturnType to match actual signature
export type ToastFunction = typeof toastFn;

/**
 * Creates a copy button action for toasts
 */
function createCopyAction(textToCopy: string) {
    return (
        <button
            onClick={() => {
                navigator.clipboard.writeText(textToCopy);
                toastFn({
                    title: "Copied!",
                    description: "Error message copied to clipboard",
                    duration: 2000,
                });
            }}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md border border-muted/40 bg-transparent px-3 text-sm font-medium hover:bg-destructive/10 focus:outline-none focus:ring-2 focus:ring-ring"
        >
            Copy
        </button>
    );
}

/**
 * Show an error toast with proper formatting for ParsedError
 * Includes hint as part of description if available
 * ALL ERROR TOASTS HAVE A COPY BUTTON
 */
export function showErrorToast(toast: ToastFunction, error: ParsedError) {
    const description = error.hint
        ? `${error.message}\n\n💡 Tip: ${error.hint}`
        : error.message;

    const fullErrorText = `Error\n\n${description}`;

    toast({
        title: 'Error',
        description,
        variant: 'destructive',
        duration: 10000, // 10 seconds for errors
        action: createCopyAction(fullErrorText),
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
 * ALL ERROR TOASTS HAVE A COPY BUTTON
 */
export function showValidationErrorToast(toast: ToastFunction, error: ParsedError) {
    if (error.fieldErrors && error.fieldErrors.length > 0) {
        const fieldCount = error.fieldErrors.length;
        const firstError = error.fieldErrors[0];

        const description = fieldCount === 1
            ? `${firstError.field}: ${firstError.message}${firstError.hint ? `\n💡 ${firstError.hint}` : ''}`
            : `${fieldCount} validation errors. First: ${firstError.field}: ${firstError.message}`;

        const fullErrorText = `Validation Error\n\n${description}`;

        toast({
            title: 'Validation Error',
            description,
            variant: 'destructive',
            duration: 10000,
            action: createCopyAction(fullErrorText),
        });
    } else {
        showErrorToast(toast, error);
    }
}
