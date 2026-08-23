import { useMemo, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import type { VLMFrameAnnotation } from '@/types/annotations';
import { getVlmAnnotationAtTime } from '@/lib/parsers/vlm';

interface VlmAnnotationPanelProps {
  annotations: VLMFrameAnnotation[];
  currentTime: number;
}

const labelVariant = (label: string): 'default' | 'secondary' | 'destructive' => {
  if (label.startsWith('ERROR')) return 'destructive';
  if (label === 'NO_TOUCH' || label === 'NO') return 'secondary';
  return 'default';
};

export const VlmAnnotationPanel = ({ annotations, currentTime }: VlmAnnotationPanelProps) => {
  const [expanded, setExpanded] = useState(false);

  const current = useMemo(
    () => getVlmAnnotationAtTime(annotations, currentTime),
    [annotations, currentTime]
  );

  if (annotations.length === 0) {
    return null;
  }

  if (!current) {
    return (
      <Card className="flex-shrink-0 p-3 text-xs text-muted-foreground">
        No VLM annotation near the current time (nearest sample points are more than a few
        seconds away).
      </Card>
    );
  }

  const hasReasoning = current.reasoning && current.reasoning !== current.label;

  return (
    <Card className="flex-shrink-0 p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Badge variant={labelVariant(current.label)}>{current.label}</Badge>
          <span className="text-xs text-muted-foreground">
            @ {current.timestamp_sec.toFixed(1)}s
            {current.sampling_mode === 'frame_burst' && current.context_frame_offsets
              ? ` · burst[${current.context_frame_offsets.join(',')}]`
              : ' · single frame'}
          </span>
        </div>
        <span className="text-[11px] text-muted-foreground whitespace-nowrap">
          {current.model}
        </span>
      </div>

      {hasReasoning && (
        <div>
          <button
            type="button"
            className="flex items-center gap-1 text-xs text-primary hover:underline"
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            {expanded ? 'Hide reasoning' : 'Show reasoning'}
          </button>
          {expanded && (
            <p className="mt-1 text-xs text-muted-foreground whitespace-pre-wrap max-h-40 overflow-y-auto">
              {current.reasoning}
            </p>
          )}
        </div>
      )}
    </Card>
  );
};
