// The pipeline guide (VideoAnnotator spec 026): what a pipeline finds, in a
// researcher's words, and where it comes from.
import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import type { PipelineDescriptor } from '@/types/pipelines';

/** The behavioural title, with the method's own name beside it when they differ. */
export const PipelineTitle = ({ pipeline }: { pipeline: PipelineDescriptor }) => {
  const title = pipeline.title ?? pipeline.name;
  return (
    <span className="flex flex-wrap items-baseline gap-x-2">
      <span className="text-sm font-medium text-foreground">{title}</span>
      {pipeline.title && pipeline.title !== pipeline.name && (
        <span className="text-xs text-muted-foreground">{pipeline.name}</span>
      )}
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
                : {b.measure} {b.figure}
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
