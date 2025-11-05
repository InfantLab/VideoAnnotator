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
    const [isLoading, setIsLoading] = useState<boolean>(false); // Start false - no auto-fetch
    const [error, setError] = useState<Error | null>(null);
    const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

    /**
     * Fetch server capabilities from health endpoint
     * Uses functional state updates to avoid stale closure issues with React StrictMode
     */
    const refresh = useCallback(async () => {
        console.log('[ServerCapabilities] refresh() called - START');

        setIsLoading(true);
        setError(null);

        try {
            console.log('[ServerCapabilities] Calling apiClient.getEnhancedHealth()...');
            const healthResponse = await apiClient.getEnhancedHealth();
            console.log('[ServerCapabilities] Health response received:', healthResponse);

            const detectedCapabilities = detectServerCapabilities(healthResponse);
            console.log('[ServerCapabilities] Capabilities detected:', detectedCapabilities);

            setCapabilities(detectedCapabilities);
            setLastRefresh(new Date());
            setError(null);
            console.log('[ServerCapabilities] SUCCESS - capabilities set');
        } catch (err) {
            console.error('[ServerCapabilities] ERROR caught:', err);
            const error = err instanceof Error ? err : new Error('Failed to detect server capabilities');
            setError(error);

            // Keep stale capabilities on refresh error (better UX than clearing)
            setCapabilities((prev) => {
                console.log('[ServerCapabilities] ERROR - keeping previous capabilities:', prev);
                return prev;
            });
        } finally {
            setIsLoading(false);
            console.log('[ServerCapabilities] refresh() called - END');
        }
    }, []); // Empty deps is safe now - we use functional updates and don't read state

    // Auto-refresh with initial delay to avoid React StrictMode double-mount issues
    useEffect(() => {
        console.log('[ServerCapabilities] useEffect triggered', {
            autoRefreshInterval,
        });

        // Initial fetch after short delay (avoids StrictMode double-fetch)
        const initialTimeout = setTimeout(() => {
            console.log('[ServerCapabilities] AUTO-REFRESH: Calling refresh() from initial timeout');
            refresh();
        }, 500); // 500ms delay

        // Set up periodic refresh
        let intervalId: NodeJS.Timeout | null = null;

        if (autoRefreshInterval && autoRefreshInterval > 0) {
            intervalId = setInterval(() => {
                console.log('[ServerCapabilities] AUTO-REFRESH: Calling refresh() from interval');
                refresh();
            }, autoRefreshInterval);
        }

        return () => {
            console.log('[ServerCapabilities] useEffect cleanup');
            clearTimeout(initialTimeout);
            if (intervalId) clearInterval(intervalId);
        };
    }, [autoRefreshInterval, refresh]); // Don't depend on capabilities - let refresh() handle it

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
