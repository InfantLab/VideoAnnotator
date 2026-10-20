import type { ReactNode } from 'react';

import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { GLOSSARY, type GlossaryWord } from '@/lib/glossary';

/** A vocabulary word with its meaning on hover or focus (spec 029 FR-023). */
export function Term({ word, children }: { word: GlossaryWord; children?: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className="cursor-help underline decoration-dotted decoration-muted-foreground underline-offset-4"
        >
          {children ?? word}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{GLOSSARY[word]}</TooltipContent>
    </Tooltip>
  );
}
