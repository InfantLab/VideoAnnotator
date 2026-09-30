/**
 * @file React context for providing pipeline catalog data throughout the app.
 * 
 * This context makes pipeline information, server capabilities, and related
 * utilities available to any component in the component tree.
 */

import { createContext, useContext } from 'react';
import type {
    PipelineCatalog,
    PipelineDescriptor,
    VideoAnnotatorServerInfo,
    VideoAnnotatorFeatureFlags,
} from '@/types/pipelines';

export interface PipelineContextValue {
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
    getPipeline: (pipelineId: string) => PipelineDescriptor | undefined;
    getFeatureFlag: (flag: keyof VideoAnnotatorFeatureFlags) => boolean;
}

export const PipelineContext = createContext<PipelineContextValue | undefined>(undefined);

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