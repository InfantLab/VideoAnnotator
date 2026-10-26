// The pipeline guide (VideoAnnotator spec 026): what a pipeline finds, in a
// researcher's words, and where it comes from.
import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { taskOfTitle } from '@/hooks/usePipelineNames';
import { kindColour, kindOfPipeline } from '@/lib/resultKinds';
import type { PipelineDescriptor } from '@/types/pipelines';

/** What a benchmark measure means, on hover (UAT #19). */
const MEASURES: Record<string, string> = {
  'DER %': 'Diarization error rate: the share of time with the wrong speaker, a missed or a false turn. Lower is better.',
  AP: 'Average precision: how well detections match the true boxes, from 0 to 1. Higher is better.',
  'mAP pose 50-95 (e2e)': 'Mean average precision for body keypoints, over strict to loose match thresholds. Higher is better.',
  MAE: 'Mean absolute error: the average size of the error. Lower is better.',
  'landmark NME (inter-ocular)': 'Landmark error, as a share of the distance between the eyes. Lower is better.',
  'action unit F1': 'F1 score for detecting facial action units, from 0 to 1. Higher is better.',
};

/** What runs, then what it finds: "pyannote community-1 · who spoke when", with its kind's icon (D1, UAT #19). */
export const PipelineTitle = ({ pipeline }: { pipeline: PipelineDescriptor }) => {
  const kind = kindOfPipeline(pipeline.id);
  const Icon = kind?.icon;
  const task = taskOfTitle(pipeline.title);
  return (
    <span className="flex flex-wrap items-center gap-x-2">
      {Icon && <Icon className="h-4 w-4 shrink-0" style={{ color: kindColour(kind) }} aria-hidden="true" />}
      <span className="text-sm font-medium text-foreground">{pipeline.name}</span>
      {task && <span className="text-xs text-muted-foreground">{task}</span>}
    </span>
  );
};

const CopyBibtex = ({ bibtex }: { bibtex: string }) => {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(bibtex);
      setState('copied');
      setTimeout(() => setState('idle'), 2000);
    } catch {
      setState('failed');
    }
  };
  return (
    <Button type="button" variant="outline" size="sm" className="h-6 px-2 text-[11px]" onClick={copy}>
      {state === 'copied' ? 'Copied' : state === 'failed' ? 'Copy failed' : 'Copy BibTeX'}
    </Button>
  );
};

/** Where a pipeline comes from and how well it does, as published (spec 026). */
export const PipelineAbout = ({
  pipeline,
  defaultOpen = false
}: {
  pipeline: PipelineDescriptor;
  defaultOpen?: boolean;
}) => {
  const { homepage, benchmarks, qualityNote, citations } = pipeline;
  if (!homepage && benchmarks === undefined && !qualityNote && !citations?.length) return null;
  return (
    <Collapsible defaultOpen={defaultOpen} className="space-y-1">
      <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
        <ChevronRight className="h-3 w-3 transition-transform [[data-state=open]_&]:rotate-90" />
        About
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-1 pl-4 text-xs text-muted-foreground">
        {homepage && (
          <p>
            <a className="underline" href={homepage} target="_blank" rel="noopener noreferrer">
              Homepage
            </a>
          </p>
        )}
        {benchmarks?.length === 0 && <p>No published benchmark</p>}
        {benchmarks && benchmarks.length > 0 && (
          <ul className="space-y-0.5">
            {benchmarks.map((b) => (
              <li key={`${b.name}-${b.measure}`}>
                <a className="underline" href={b.source} target="_blank" rel="noopener noreferrer">
                  {b.name}
                </a>
                : {MEASURES[b.measure] ? <abbr title={MEASURES[b.measure]} className="cursor-help">{b.measure}</abbr> : b.measure} {b.figure}
                {b.note && <span className="text-muted-foreground/80"> ({b.note})</span>}
              </li>
            ))}
          </ul>
        )}
        {qualityNote && <p>{qualityNote}</p>}
        {citations && citations.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span>Cite: {citations.map((c) => c.key).join(', ')}</span>
            <CopyBibtex bibtex={citations.map((c) => c.bibtex).join('\n\n')} />
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
};
