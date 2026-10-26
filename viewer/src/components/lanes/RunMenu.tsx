import { Check, ChevronDown } from 'lucide-react';

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { Lane } from '@/lib/lanes';
import type { RunOption } from '@/lib/videoRuns';
import { formatWhen } from '@/lib/when';

/** What a lane's run menu can do (spec 028 US5); absent for local files. */
export interface RunControls {
  options: (pipeline: string) => RunOption[];
  isCompare: (pipeline: string, jobId: string) => boolean;
  choose: (pipeline: string, jobId: string) => void;
  compare: (pipeline: string, jobId: string) => void;
  removeCompare: (pipeline: string, jobId: string) => void;
}

const when = (o: RunOption) => formatWhen(o.date);

/** The run a lane shows, another run, or a comparison below it (spec 028 FR-016). */
export function RunMenu({ lane, controls }: { lane: Lane; controls: RunControls }) {
  const pipeline = lane.pipeline;
  const current = lane.run?.id;
  if (!pipeline || !current) return null;
  const options = controls.options(pipeline);
  const comparing = controls.isCompare(pipeline, current);
  const others = options.filter((o) => o.usable && o.id !== current && !controls.isCompare(pipeline, o.id));
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="absolute right-1 top-0 z-[1] flex items-center gap-0.5 rounded bg-card/80 px-1 text-[10px] text-muted-foreground hover:text-foreground"
          title={`${lane.label} from ${lane.run!.label}: choose another run, or compare`}
          aria-label={`Run shown in ${lane.label}: ${lane.run!.label}`}
          onClick={(e) => e.stopPropagation()}
          data-testid="lane-run"
        >
          {comparing && <span className="font-medium">compared: </span>}
          {lane.run!.label}
          <ChevronDown className="h-3 w-3" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-72 p-2 text-xs" align="end" data-testid="run-menu" onClick={(e) => e.stopPropagation()}>
        {comparing ? (
          <button
            type="button"
            className="w-full rounded px-2 py-1 text-left hover:bg-muted"
            onClick={() => controls.removeCompare(pipeline, current)}
          >
            Remove this comparison
          </button>
        ) : (
          <>
            <p className="mb-1 font-medium">Show the results of</p>
            <ul>
              {options.map((o) => (
                <li key={o.id}>
                  <button
                    type="button"
                    className="flex w-full items-start gap-1 rounded px-2 py-1 text-left hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={!o.usable}
                    title={o.usable ? undefined : o.error ?? o.status}
                    aria-current={o.id === current}
                    onClick={() => controls.choose(pipeline, o.id)}
                  >
                    <Check className={o.id === current ? 'mt-0.5 h-3 w-3 shrink-0' : 'mt-0.5 h-3 w-3 shrink-0 opacity-0'} />
                    <span className="min-w-0">
                      <span className="block truncate">{o.label}</span>
                      <span className="block text-muted-foreground">
                        {when(o)} · {o.usable ? o.status : `${o.status}${o.error ? `: ${o.error}` : ''}`}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {others.length > 0 && (
              <>
                <p className="mb-1 mt-2 font-medium">Compare with…</p>
                <ul>
                  {others.map((o) => (
                    <li key={o.id}>
                      <button
                        type="button"
                        className="w-full truncate rounded px-2 py-1 text-left hover:bg-muted"
                        onClick={() => controls.compare(pipeline, o.id)}
                      >
                        {o.label} <span className="text-muted-foreground">{when(o)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
