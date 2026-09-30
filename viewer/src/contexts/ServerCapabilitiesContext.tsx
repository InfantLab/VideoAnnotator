/**
 * React Context for sharing server capabilities across the application
 * 
 * Provides:
 * - Server capabilities (version, feature flags)
 * - Authentication status
 * - Manual refresh mechanism
 * - Loading/error states
 */

import { createContext, useContext } from 'react';
import type { ServerCapabilities } from '@/types/api';

export interface ServerCapabilitiesContextValue {
    capabilities: ServerCapabilities | null;
    isLoading: boolean;
    error: Error | null;
    lastRefresh: Date | null;
    refresh: () => Promise<void>;
}

export const ServerCapabilitiesContext = createContext<ServerCapabilitiesContextValue | undefined>(
    undefined
);

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
