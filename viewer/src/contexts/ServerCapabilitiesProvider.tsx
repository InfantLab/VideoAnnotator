import React, { useCallback, useEffect, useState } from 'react';
import type { ServerCapabilities } from '@/types/api';
import { detectServerCapabilities } from '@/api/capabilities';
import { apiClient } from '@/api/client';
import {
  ServerCapabilitiesContext,
  type ServerCapabilitiesContextValue,
} from './ServerCapabilitiesContext';

interface ServerCapabilitiesProviderProps {
  children: React.ReactNode;
  autoRefreshInterval?: number; // milliseconds, default 2 minutes
}

/**
 * Provider component that fetches and shares server capabilities.
 */
export function ServerCapabilitiesProvider({
  children,
  autoRefreshInterval = 2 * 60 * 1000,
}: ServerCapabilitiesProviderProps) {
  const [capabilities, setCapabilities] = useState<ServerCapabilities | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

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
      const nextError = err instanceof Error ? err : new Error('Failed to detect server capabilities');
      setError(nextError);

      // Keep stale capabilities on refresh error
      setCapabilities((prev) => prev);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialTimeout = setTimeout(() => {
      refresh();
    }, 500);

    let intervalId: ReturnType<typeof setInterval> | null = null;
    if (autoRefreshInterval && autoRefreshInterval > 0) {
      intervalId = setInterval(() => {
        refresh();
      }, autoRefreshInterval);
    }

    return () => {
      clearTimeout(initialTimeout);
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefreshInterval, refresh]);

  const value: ServerCapabilitiesContextValue = {
    capabilities,
    isLoading,
    error,
    lastRefresh,
    refresh,
  };

  return <ServerCapabilitiesContext.Provider value={value}>{children}</ServerCapabilitiesContext.Provider>;
}
