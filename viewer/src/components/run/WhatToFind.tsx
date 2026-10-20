import { useEffect, useMemo, type ReactNode } from 'react';
import { useQueries } from '@tanstack/react-query';

import { ConfigValidationPanel } from '@/components/ConfigValidationPanel';
import { DynamicPipelineParameters } from '@/components/DynamicPipelineParameters';
import { PipelineSelectionStep } from '@/components/run/PipelineChoice';
import { pipelineSchemaQueryOptions, useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';
import { useConfigValidation } from '@/hooks/useConfigValidation';
import type { PipelineDescriptor } from '@/types/pipelines';

interface WhatToFindProps {
  pipelines: PipelineDescriptor[];
  isLoading: boolean;
  error: unknown;
  restartRequired: boolean;
  selected: string[];
  onSelect: (pipelines: string[]) => void;
  config: Record<string, unknown>;
  onConfig: (updater: (prev: Record<string, unknown>) => Record<string, unknown>) => void;
  /** "Settings from: this project's settings, last changed 12 Oct". */
  source: ReactNode;
  /** The workflow picker and Save as workflow. */
  workflows?: ReactNode;
  notices?: string[];
  onDismissNotices?: () => void;
  /** Reports whether the settings can be used (validation, URL fields). */
  onValidity?: (valid: boolean) => void;
}

/** The pipelines to run, with their guide, and their settings (spec 029 R9). */
export function WhatToFind({
  pipelines,
  isLoading,
  error,
  restartRequired,
  selected,
  onSelect,
  config,
  onConfig,
  source,
  workflows,
  notices,
  onDismissNotices,
  onValidity,
}: WhatToFindProps) {
  const refresh = useRefreshPipelineCatalog();
  const { validationResult, isValidating, validateConfig } = useConfigValidation();
  // The catalog has no per-field settings; each selected pipeline's schema has.
  const schemas = useQueries({ queries: selected.map((id) => pipelineSchemaQueryOptions(id)) });
  const withSchemas = useMemo(() => {
    const byId = new Map(selected.map((id, i) => [id, schemas[i]?.data?.parameters] as const));
    return pipelines.map((p) => (byId.get(p.id) ? { ...p, parameters: byId.get(p.id)! } : p));
  }, [pipelines, selected, schemas]);

  useEffect(() => {
    if (Object.keys(config).length > 0) validateConfig(config);
  }, [config, validateConfig]);
  useEffect(() => {
    onValidity?.(validationResult?.valid !== false);
  }, [validationResult, onValidity]);

  return (
    <div className="space-y-6">
      <p className="text-sm" data-testid="settings-source">
        <span className="text-muted-foreground">Settings from:</span> {source}
      </p>
      <PipelineSelectionStep
        pipelines={pipelines}
        selectedPipelines={selected}
        setSelectedPipelines={onSelect}
        isLoading={isLoading}
        error={error}
        onRetry={() => refresh({ forceServerRefresh: true })}
        restartRequired={restartRequired}
        selectionNotices={notices}
        onDismissNotices={onDismissNotices}
        presetBar={workflows}
      />
      {selected.length > 0 && (
        <section className="space-y-3">
          <h3 className="font-medium">Settings</h3>
          <p className="text-sm text-muted-foreground">
            Each pipeline&apos;s main settings; hover a name for what it does and its default. Change only what your
            study needs.
          </p>
          <DynamicPipelineParameters pipelines={withSchemas} selectedPipelineIds={selected} config={config} onConfigChange={onConfig} />
          <ConfigValidationPanel validationResult={validationResult} isValidating={isValidating} />
        </section>
      )}
    </div>
  );
}
