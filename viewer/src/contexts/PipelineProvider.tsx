import React from 'react';
import { usePipelineData, usePipelineUtils } from '@/hooks/usePipelineData';
import type { PipelineDescriptor, VideoAnnotatorFeatureFlags } from '@/types/pipelines';
import { PipelineContext, type PipelineContextValue } from './PipelineContext';

interface PipelineProviderProps {
  children: React.ReactNode;
}

/**
 * Provider component that provides pipeline data context to child components.
 * Data is NOT fetched automatically - components must call refetch() when needed.
 */
export function PipelineProvider({ children }: PipelineProviderProps) {
  // enabled: false means data is NOT fetched on mount - lazy loading
  const pipelineData = usePipelineData({ enabled: false });
  const pipelineUtils = usePipelineUtils();

  const isPipelineAvailable = (pipelineId: string): boolean => {
    return !!pipelineData.catalog?.pipelines.find((pipeline) => pipeline.id === pipelineId);
  };

  const getPipeline = (pipelineId: string): PipelineDescriptor | undefined => {
    return pipelineData.catalog?.pipelines.find((pipeline) => pipeline.id === pipelineId);
  };

  const getFeatureFlag = (flag: keyof VideoAnnotatorFeatureFlags): boolean => {
    return pipelineData.server?.features?.[flag] ?? false;
  };

  const contextValue: PipelineContextValue = {
    catalog: pipelineData.catalog,
    server: pipelineData.server,
    features: pipelineData.server?.features ?? {},

    isLoading: pipelineData.isLoading,
    isError: pipelineData.isError,
    error: pipelineData.error as Error | null,

    refetch: pipelineData.refetch,
    refreshCatalog: pipelineUtils.refreshCatalog,
    refreshServerInfo: pipelineUtils.refreshServerInfo,
    clearAllCache: pipelineUtils.clearAllCache,

    isPipelineAvailable,
    getPipeline,
    getFeatureFlag,
  };

  return <PipelineContext.Provider value={contextValue}>{children}</PipelineContext.Provider>;
}
