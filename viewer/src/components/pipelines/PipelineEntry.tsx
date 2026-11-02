import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronRight, Play } from 'lucide-react';

import { PipelineSettings } from '@/components/DynamicPipelineParameters';
import { ReadinessDetails } from '@/components/LockedPipelineCard';
import { PipelineAbout, PipelineTitle } from '@/components/PipelineGuide';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { usePipelineSchema } from '@/hooks/usePipelineCatalog';
import { pipelineSummary } from '@/lib/pipelineGuide';
import { getPipelineConfig, settingsSummary } from '@/lib/pipelineSettings';
import { kindOfPipeline } from '@/lib/resultKinds';
import { runState } from '@/lib/runTarget';
import type { PipelineDescriptor } from '@/types/pipelines';

type Updater = (prev: Record<string, unknown>) => Record<string, unknown>;

/** "Shows as the Speech lane on the Video page; files: JSON, WebVTT." */
function whatItWrites(pipeline: PipelineDescriptor): string | null {
  const kind = kindOfPipeline(pipeline.id);
  const formats = pipeline.outputFormats ?? [];
  const parts = [
    kind ? `Shows as the ${kind.label} lane on the Video page` : null,
    formats.length ? `files: ${formats.join(', ')}` : null,
  ].filter(Boolean);
  return parts.length ? `${parts.join('; ')}.` : null;
}

/**
 * A ready pipeline on the Pipelines page (UAT #71): what it finds, your own
 * defaults for its settings, where it comes from, and a run with it.
 */
export function PipelineEntry({
  pipeline,
  yours,
  onYours,
}: {
  pipeline: PipelineDescriptor;
  /** Your defaults, keyed by pipeline, as a run's `config` is. */
  yours: Record<string, unknown>;
  onYours: (updater: Updater) => void;
}) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const schema = usePipelineSchema(pipeline.id);
  const withSchema = schema.data ? { ...pipeline, parameters: schema.data.parameters } : pipeline;
  const cue = withSchema.parameters ? settingsSummary(withSchema, yours) : '';
  const own = getPipelineConfig(yours, pipeline.id);
  const hasOwn = Object.keys(own).length > 0;
  const writes = whatItWrites(pipeline);

  const runIt = () =>
    navigate('/run', {
      state: runState({
        target: null,
        settings: {
          selectedPipelines: [pipeline.id],
          config: hasOwn ? { [pipeline.id]: own } : {},
          label: hasOwn ? `your defaults for ${pipeline.name}` : `${pipeline.name}'s defaults`,
        },
      }),
    });

  return (
    <div className="rounded-lg border border-border bg-card" data-testid="pipeline-entry">
      <button
        type="button"
        className="flex w-full flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-2 text-left"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <PipelineTitle pipeline={pipeline} />
        <span className="ml-auto flex min-w-0 items-center gap-2">
          {cue && cue !== 'defaults' && (
            <span className="truncate text-xs text-muted-foreground" data-testid="your-defaults-cue">
              Yours: <span className="text-foreground">{cue}</span>
            </span>
          )}
          <Badge className="bg-green-600 px-1.5 py-0 text-[10px] hover:bg-green-600">Ready</Badge>
          <ChevronRight className={`h-4 w-4 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`} aria-hidden="true" />
        </span>
      </button>
      {open && (
        <div className="space-y-4 border-t px-4 py-3">
          <div className="space-y-1 text-sm">
            <p>{pipelineSummary(pipeline)}</p>
            {writes && <p className="text-muted-foreground">{writes}</p>}
            <ReadinessDetails notes={pipeline.readiness?.notes} />
          </div>
          {withSchema.parameters && withSchema.parameters.length > 0 && (
            <section className="space-y-1" aria-label={`${pipeline.name} settings`}>
              <h3 className="text-sm font-semibold">Your defaults</h3>
              <p className="text-xs text-muted-foreground">
                New run starts from these wherever a project, a workflow or your last run doesn&apos;t set the
                setting, and says so. Saved as you change them; ● marks what differs from VideoAnnotator&apos;s default.
              </p>
              <div className="divide-y">
                <PipelineSettings pipeline={withSchema} config={yours} onConfigChange={onYours} describe />
              </div>
            </section>
          )}
          <div className="flex flex-wrap items-end justify-between gap-2">
            <PipelineAbout pipeline={pipeline} />
            <span className="flex items-center gap-2">
              {pipeline.version && <span className="text-xs text-muted-foreground">v{pipeline.version}</span>}
              <Button size="sm" variant="outline" onClick={runIt}>
                <Play className="mr-1 h-4 w-4" />
                Run on videos…
              </Button>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
