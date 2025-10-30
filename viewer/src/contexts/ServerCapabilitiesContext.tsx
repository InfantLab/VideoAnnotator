/**
 * React Context for sharing server capabilities across the application
 * 
 * Provides:
 * - Server capabilities (version, feature flags)
 * - Authentication status
 * - Manual refresh mechanism
 * - Loading/error states
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { ServerCapabilities } from '@/types/api';
import { detectServerCapabilities } from '@/api/capabilities';
import { apiClient } from '@/api/client';

interface ServerCapabilitiesContextValue {
    capabilities: ServerCapabilities | null;
    isLoading: boolean;
    error: Error | null;
    lastRefresh: Date | null;
    refresh: () => Promise<void>;
}

const ServerCapabilitiesContext = createContext<ServerCapabilitiesContextValue | undefined>(
    undefined
);

interface ServerCapabilitiesProviderProps {
    children: React.ReactNode;
    autoRefreshInterval?: number; // milliseconds, default 2 minutes
}

/**
 * Provider component that fetches and shares server capabilities
 * 
 * Usage:
 * ```tsx
 * <ServerCapabilitiesProvider>
 *   <App />
 * </ServerCapabilitiesProvider>
 * ```
 */
export function ServerCapabilitiesProvider({
    children,
    autoRefreshInterval = 2 * 60 * 1000, // 2 minutes default
}: ServerCapabilitiesProviderProps) {
    const [capabilities, setCapabilities] = useState<ServerCapabilities | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<Error | null>(null);
    const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

    /**
     * Fetch server capabilities from health endpoint
     */
    const refresh = useCallback(async () => {
        setIsLoading(true);
        setError(null);

        try {
            const healthResponse = await apiClient.getEnhancedHealth();
            const detectedCapabilities = detectServerCapabilities(healthResponse);

            setCapabilities(detectedCapabilities);
            setLastRefresh(new Date());
            setError(null);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to detect server capabilities');
            setError(error);

            // Keep stale capabilities on refresh error (better UX than clearing)
            // setCapabilities to null only on initial load failure
            setCapabilities((prev) => prev || null);
        } finally {
            setIsLoading(false);
        }
    }, []); // Remove capabilities from dependencies to prevent infinite loop

    // Initial fetch on mount
    useEffect(() => {
        refresh();
    }, [refresh]);

    // Auto-refresh interval - use faster polling when there's an error for quick recovery
    useEffect(() => {
        if (!autoRefreshInterval || autoRefreshInterval <= 0) {
            return;
        }

        // When there's an error, poll more frequently (every 10 seconds) for faster recovery
        const effectiveInterval = error ? 10000 : autoRefreshInterval;

        const intervalId = setInterval(() => {
            refresh();
        }, effectiveInterval);

        return () => clearInterval(intervalId);
    }, [autoRefreshInterval, refresh, error]);

    const value: ServerCapabilitiesContextValue = {
        capabilities,
        isLoading,
        error,
        lastRefresh,
        refresh,
    };

    return (
        <ServerCapabilitiesContext.Provider value={value}>
            {children}
        </ServerCapabilitiesContext.Provider>
    );
}

/**
 * Hook to access server capabilities context
 * 
 * @throws Error if used outside ServerCapabilitiesProvider
 * 
 * Usage:
 * ```tsx
 * const { capabilities, refresh, isLoading } = useServerCapabilities();
 * 
 * if (capabilities?.supportsJobCancellation) {
 *   // Show cancel button
 * }
 * ```
 */
export function useServerCapabilitiesContext(): ServerCapabilitiesContextValue {
    const context = useContext(ServerCapabilitiesContext);

    if (context === undefined) {
        throw new Error(
            'useServerCapabilitiesContext must be used within a ServerCapabilitiesProvider'
        );
    }

    return context;
}

/**
 * Hook to get server version string
 * 
 * @returns Server version or 'unknown' if not available
 */
export function useServerVersion(): string {
    const { capabilities } = useServerCapabilitiesContext();
    return capabilities?.version || 'unknown';
}

/**
 * Hook to check if server supports a specific feature
 * 
 * @param feature - Feature flag to check
 * @returns true if feature is supported, false otherwise
 */
export function useServerFeature(
    feature: keyof Omit<ServerCapabilities, 'version' | 'detectedAt'>
): boolean {
    const { capabilities } = useServerCapabilitiesContext();

    if (!capabilities) {
        return false;
    }

    return capabilities[feature] as boolean;
}
