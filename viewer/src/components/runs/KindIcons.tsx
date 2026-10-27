import { usePipelineNames } from '@/hooks/usePipelineNames';
import { kindColour, kindOfPipeline } from '@/lib/resultKinds';

/** What a run looks for, as its kinds' icons in their colours, names on hover (viewer design "Run row"). */
export function KindIcons({ pipelines }: { pipelines: string[] }) {
  const { nameOf, order } = usePipelineNames();
  const sorted = [...pipelines].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  return (
    <span className="inline-flex shrink-0 items-center gap-1" data-testid="run-kinds">
      {sorted.map((pipeline) => {
        const kind = kindOfPipeline(pipeline);
        if (!kind) return null;
        const Icon = kind.icon;
        return (
          <span key={pipeline} title={nameOf(pipeline)} aria-label={nameOf(pipeline)} role="img" className="inline-flex">
            <Icon className="h-4 w-4" style={{ color: kindColour(kind) }} aria-hidden="true" />
          </span>
        );
      })}
    </span>
  );
}
