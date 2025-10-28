// React hook for configuration validation with debouncing and caching
// Provides real-time validation feedback with 500ms debounce and config hash caching

import { useState, useEffect, useCallback, useRef } from 'react';
import { apiClient } from '@/api/client';
import type { ConfigValidationResult } from '@/types/api';

/**
 * Generate stable hash for config object (order-independent)
 */
function hashConfig(config: Record<string, any>): string {
    // Sort keys recursively for order-independent hash
    const normalize = (obj: any): any => {
        if (obj === null || typeof obj !== 'object') return obj;
        if (Array.isArray(obj)) return obj.map(normalize);
        return Object.keys(obj)
            .sort()
            .reduce((acc, key) => {
                acc[key] = normalize(obj[key]);
                return acc;
            }, {} as Record<string, any>);
    };

    const normalized = normalize(config);
    return JSON.stringify(normalized);
}

interface UseConfigValidationReturn {
    validationResult: ConfigValidationResult | null;
    isValidating: boolean;
    error: Error | null;
    validateConfig: (config: Record<string, any>) => void;
    validateNow: (config: Record<string, any>) => Promise<void>;
    clearValidation: () => void;
}

/**
 * Hook for real-time configuration validation with debouncing
 * 
 * Features:
 * - 500ms debounce on config changes
 * - Config hash caching (avoids redundant API calls)
 * - Loading state tracking
 * - Error handling
 * 
 * @returns Validation state and control functions
 */
export function useConfigValidation(): UseConfigValidationReturn {
    const [validationResult, setValidationResult] = useState<ConfigValidationResult | null>(null);
    const [isValidating, setIsValidating] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
    const cacheRef = useRef<Map<string, ConfigValidationResult>>(new Map());
    const abortControllerRef = useRef<AbortController | null>(null);

    /**
     * Perform validation (internal)
     */
    const performValidation = useCallback(async (config: Record<string, any>) => {
        // Cancel any pending validation
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        const configHash = hashConfig(config);

        // Check cache first
        const cached = cacheRef.current.get(configHash);
        if (cached) {
            setValidationResult(cached);
            setIsValidating(false);
            setError(null);
            return;
        }

        // Start validation
        setIsValidating(true);
        setError(null);

        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        try {
            const result = await apiClient.validateConfig(config);

            // Only update if this request wasn't aborted
            if (!abortController.signal.aborted) {
                setValidationResult(result);
                cacheRef.current.set(configHash, result);
                setError(null);
            }
        } catch (err) {
            if (!abortController.signal.aborted) {
                const error = err instanceof Error ? err : new Error('Validation failed');
                setError(error);
                setValidationResult(null);
            }
        } finally {
            if (!abortController.signal.aborted) {
                setIsValidating(false);
            }
        }
    }, []);

    /**
     * Validate config with debouncing
     */
    const validateConfig = useCallback((config: Record<string, any>) => {
        // Clear existing timer
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
        }

        // Set new timer for 500ms
        debounceTimerRef.current = setTimeout(() => {
            performValidation(config);
        }, 500);
    }, [performValidation]);

    /**
     * Validate immediately without debounce
     */
    const validateNow = useCallback(async (config: Record<string, any>) => {
        // Clear any pending debounced validation
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
            debounceTimerRef.current = null;
        }

        await performValidation(config);
    }, [performValidation]);

    /**
     * Clear validation state
     */
    const clearValidation = useCallback(() => {
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
            debounceTimerRef.current = null;
        }
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        setValidationResult(null);
        setIsValidating(false);
        setError(null);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    return {
        validationResult,
        isValidating,
        error,
        validateConfig,
        validateNow,
        clearValidation,
    };
}
