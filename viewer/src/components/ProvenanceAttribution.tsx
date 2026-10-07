import { Info } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { attributionLabel } from '@/lib/provenance';
import type { ProvenanceInfo } from '@/types/annotations';

interface ProvenanceAttributionProps {
  info: ProvenanceInfo | undefined;
  /** Shown when the file doesn't name its pipeline. */
  pipeline: string;
}

const Field = ({ label, value }: { label: string; value: unknown }) =>
  value === undefined || value === null || value === '' ? null : (
    <div className="grid grid-cols-3 gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="col-span-2 break-all font-mono">{typeof value === 'string' ? value : JSON.stringify(value)}</dd>
    </div>
  );

/**
 * Which pipeline and version drew an overlay (constitution Principle VI), with
 * the full record on demand. Values are shown as recorded, never filled in.
 */
export const ProvenanceAttribution = ({ info, pipeline }: ProvenanceAttributionProps) => {
  const label = attributionLabel(info, pipeline);
  return (
    <div className="flex items-start gap-1 text-xs text-muted-foreground">
      <span className="min-w-0">
        {/* Pipeline names are long single words in a narrow column: wrap after underscores, not mid-word. */}
        {label.split('_').map((part, i) => (
          <span key={i}>
            {i > 0 && '_'}
            {i > 0 && <wbr />}
            {part}
          </span>
        ))}
      </span>
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="shrink-0 rounded p-0.5 hover:bg-muted"
            aria-label={`Provenance: ${label}`}
          >
            <Info className="h-3 w-3" />
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-96 max-h-96 overflow-auto text-xs">
          {info?.kind === 'recorded' ? (
            <dl className="space-y-1">
              <Field label="Pipeline" value={info.record.pipeline.name} />
              <Field label="Part of" value={info.record.pipeline.sub_pipeline} />
              <Field label="VideoAnnotator" value={info.record.videoannotator_version} />
              <Field label="Created" value={info.record.created_at} />
              <Field label="Job" value={info.record.job_id} />
              <Field label="Video" value={info.record.input?.name} />
              <Field label="Video sha256" value={info.record.input?.sha256} />
              {(info.record.models ?? []).map((m, i) => (
                <Field
                  key={`${m.name}-${i}`}
                  label={i === 0 ? 'Models' : ''}
                  value={`${m.name} (${m.source}) ${m.revision ?? `revision not recorded${m.revision_note ? `: ${m.revision_note}` : ''}`}`}
                />
              ))}
              <Field label="Determinism" value={info.record.determinism} />
              <Field label="VLM" value={info.record.vlm} />
              <Field label="Settings" value={info.record.settings} />
            </dl>
          ) : info?.kind === 'ground_truth' ? (
            <p>Human-coded ground truth, loaded from {info.fileName}. Not a pipeline output.</p>
          ) : (
            <p>
              This file doesn't record what made it
              {info?.kind === 'partial' && info.videoannotatorVersion
                ? ` beyond the VideoAnnotator version (${info.videoannotatorVersion})`
                : ''}
              . Files made before VideoAnnotator 1.6, or by a pipeline run outside a job, have no record.
            </p>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
};
