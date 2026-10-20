// Results out (spec 025): one table of every pipeline's results, ELAN files,
// and a methods paragraph, for a job, a run, a project or (spec 029) any
// videos. The server builds them from the result files; this only asks and
// saves what comes back.

import { lazy, Suspense, useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Table2, Video } from 'lucide-react';
import { apiClient, type ExportResults, type ExportScope } from '@/api/client';
import { apiErrorEnvelope } from '@/api/handleError';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import { saveBlob, summaryLines } from '@/lib/exports';
import { userMessage } from '@/lib/userMessage';

const MethodsDialog = lazy(() => import('./MethodsDialog'));

type Busy = 'csv' | 'parquet' | 'elan' | null;

const FALLBACK_NAMES: Record<Exclude<Busy, null>, string> = {
  csv: 'results.csv',
  parquet: 'results.parquet',
  elan: 'results.eaf',
};

interface ExportMenuProps {
  scope: ExportScope;
  /** "Export ▾" opening a panel, for a header or a selection bar. */
  popover?: boolean;
}

/** Videos from several runs can be exported from their newest results or from every run. */
const hasRunChoice = (scope: ExportScope) => scope.kind === 'videos' || scope.kind === 'project';

/** Why Parquet isn't available, and (for an admin) a way to fix it. Asks who
 * the user is only when shown. */
function ParquetHint({ hint }: { hint: string }) {
  const { isAdmin } = useCurrentUser();
  const [installing, setInstalling] = useState<'idle' | 'started' | 'failed'>('idle');

  const installParquet = async () => {
    try {
      await apiClient.installPipelineExtras('parquet');
      setInstalling('started');
    } catch {
      setInstalling('failed');
    }
  };

  return (
    <div className="text-sm space-y-1" role="status">
      <p>
        Parquet needs an optional component on the server:{' '}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{hint}</code>. CSV works without it.
      </p>
      {isAdmin === true && installing === 'idle' && (
        <Button variant="outline" size="sm" onClick={installParquet}>
          Install it now
        </Button>
      )}
      {installing === 'started' && (
        <p className="text-muted-foreground">Installing; try Parquet again in a minute or two.</p>
      )}
      {installing === 'failed' && (
        <p className="text-muted-foreground">Couldn&apos;t start the install. Run the command above on the server.</p>
      )}
    </div>
  );
}

export function ExportMenu({ scope, popover = false }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState<ExportResults>('latest');
  const [busy, setBusy] = useState<Busy>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [parquetHint, setParquetHint] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [methodsOpen, setMethodsOpen] = useState(false);
  const datasetOnly = scope.kind === 'dataset';
  const chosen: ExportScope = hasRunChoice(scope) ? ({ ...scope, results } as ExportScope) : scope;

  const run = async (which: Exclude<Busy, null>) => {
    setBusy(which);
    setProblem(null);
    setLines([]);
    try {
      const download = await apiClient.exportResults(
        chosen,
        which === 'elan' ? 'elan' : 'table',
        which === 'parquet' ? 'parquet' : 'csv'
      );
      saveBlob(download.blob, download.filename ?? FALLBACK_NAMES[which]);
      setLines(summaryLines(download.summary));
    } catch (error) {
      const envelope = apiErrorEnvelope(error);
      if (envelope.code === 'PARQUET_NOT_INSTALLED') {
        setParquetHint(envelope.hint ?? 'pip install videoannotator[parquet]');
      } else {
        setProblem(`Couldn't export: ${userMessage(error)}`);
      }
    } finally {
      setBusy(null);
    }
  };

  const actions = (
    <>
      {hasRunChoice(scope) && (
        <RadioGroup
          value={results}
          onValueChange={(value) => setResults(value as ExportResults)}
          className="flex flex-wrap gap-4 text-sm"
          aria-label="Which results"
        >
          <div className="flex items-center gap-2">
            <RadioGroupItem value="latest" id="export-latest" />
            <Label htmlFor="export-latest" title="For each video and pipeline, its newest finished result">
              Latest results
            </Label>
          </div>
          <div className="flex items-center gap-2">
            <RadioGroupItem value="all" id="export-all" />
            <Label htmlFor="export-all" title="Every run's results; the table's run column tells them apart">
              Every run
            </Label>
          </div>
        </RadioGroup>
      )}
      <div id="export-actions" className="flex flex-wrap gap-2" role="group" aria-label="Export">
        <Button variant="secondary" size="sm" onClick={() => run('csv')} disabled={busy !== null}>
          <Table2 className="h-4 w-4 mr-2" />
          {busy === 'csv' ? 'Preparing…' : 'Results table (CSV)'}
        </Button>
        <Button variant="secondary" size="sm" onClick={() => run('parquet')} disabled={busy !== null}>
          <Table2 className="h-4 w-4 mr-2" />
          {busy === 'parquet' ? 'Preparing…' : 'Results table (Parquet)'}
        </Button>
        {!datasetOnly && (
          <>
            <Button variant="secondary" size="sm" onClick={() => run('elan')} disabled={busy !== null}>
              <Video className="h-4 w-4 mr-2" />
              {busy === 'elan' ? 'Preparing…' : 'ELAN (.eaf)'}
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setMethodsOpen(true)}>
              <FileText className="h-4 w-4 mr-2" />
              Methods
            </Button>
          </>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        One table of every pipeline&apos;s results (one row per event), ready for R or Python
        {datasetOnly ? ', covering every run of this project.' : '; ELAN files, one per video; or a methods paragraph with citations.'}
      </p>
    </>
  );

  const feedback = (
    <>
      {parquetHint && <ParquetHint hint={parquetHint} />}
      {problem && (
        <p className="text-sm text-destructive" role="alert">
          {problem}
        </p>
      )}
      {lines.length > 0 && (
        <div className="text-xs text-muted-foreground" role="status">
          <p className="font-medium">Exported, with these left out:</p>
          <ul className="list-disc pl-4">
            {lines.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}
      {methodsOpen && chosen.kind !== 'dataset' && (
        <Suspense fallback={null}>
          <MethodsDialog scope={chosen} open={methodsOpen} onOpenChange={setMethodsOpen} />
        </Suspense>
      )}
    </>
  );

  if (popover) {
    return (
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm">
            Export
            <ChevronDown className="h-4 w-4 ml-1" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-96 space-y-3" align="end" data-testid="export-panel">
          {actions}
          {feedback}
        </PopoverContent>
      </Popover>
    );
  }

  return (
    <div className="space-y-2">
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="export-actions"
      >
        {open ? <ChevronUp className="h-4 w-4 mr-2" /> : <ChevronDown className="h-4 w-4 mr-2" />}
        Export
      </Button>
      {open && actions}
      {feedback}
    </div>
  );
}
