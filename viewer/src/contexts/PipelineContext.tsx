/**
 * @file React context for providing pipeline catalog data throughout the app.
 * 
 * This context makes pipeline information, server capabilities, and related
 * utilities available to any component in the component tree.
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { usePipelineData, usePipelineUtils } from '@/hooks/usePipelineData';
import type {
    PipelineCatalog,
    VideoAnnotatorServerInfo,
    VideoAnnotatorFeatureFlags,
} from '@/types/pipelines';

interface PipelineContextValue {
    // Data
    catalog: PipelineCatalog | undefined;
    server: VideoAnnotatorServerInfo | undefined;
    features: VideoAnnotatorFeatureFlags;

    // Loading states
    isLoading: boolean;
    isError: boolean;
    error: Error | null;

    // Utilities
    refetch: () => void;
    refreshCatalog: () => Promise<void>;
    refreshServerInfo: () => Promise<void>;
    clearAllCache: () => void;

    // Convenience methods
    isPipelineAvailable: (pipelineId: string) => boolean;
    getPipeline: (pipelineId: string) => any;
    getFeatureFlag: (flag: keyof VideoAnnotatorFeatureFlags) => boolean;
}

const PipelineContext = createContext<PipelineContextValue | undefined>(undefined);

interface PipelineProviderProps {
    children: ReactNode;
}

/**
 * Provider component that provides pipeline data context to child components.
 * Data is NOT fetched automatically - components must call refetch() when needed.
 * This prevents unnecessary API calls on pages that don't need pipeline data.
 * 
 * @param props - Provider props containing children
 * @returns JSX element wrapping children with pipeline context
 */
export function PipelineProvider({ children }: PipelineProviderProps) {
    // enabled: false means data is NOT fetched on mount - lazy loading
    const pipelineData = usePipelineData({ enabled: false });
    const pipelineUtils = usePipelineUtils();

    // Helper functions
    const isPipelineAvailable = (pipelineId: string): boolean => {
        return !!pipelineData.catalog?.pipelines.find(p => p.id === pipelineId);
    };

    const getPipeline = (pipelineId: string) => {
        return pipelineData.catalog?.pipelines.find(p => p.id === pipelineId);
    };

    const getFeatureFlag = (flag: keyof VideoAnnotatorFeatureFlags): boolean => {
        return pipelineData.server?.features?.[flag] ?? false;
    };

    const contextValue: PipelineContextValue = {
        // Data
        catalog: pipelineData.catalog,
        server: pipelineData.server,
        features: pipelineData.server?.features ?? {},

        // Loading states
        isLoading: pipelineData.isLoading,
        isError: pipelineData.isError,
        error: pipelineData.error as Error | null,

        // Utilities
        refetch: pipelineData.refetch,
        refreshCatalog: pipelineUtils.refreshCatalog,
        refreshServerInfo: pipelineUtils.refreshServerInfo,
        clearAllCache: pipelineUtils.clearAllCache,

        // Convenience methods
        isPipelineAvailable,
        getPipeline,
        getFeatureFlag,
    };

    return (
        <PipelineContext.Provider value={contextValue}>
            {children}
        </PipelineContext.Provider>
    );
}

/**
 * Hook to access the pipeline context.
 * 
 * @returns Pipeline context value with data and utilities
 * @throws Error if used outside of PipelineProvider
 */
export function usePipelineContext(): PipelineContextValue {
    const context = useContext(PipelineContext);

    if (context === undefined) {
        throw new Error('usePipelineContext must be used within a PipelineProvider');
    }

    return context;
}