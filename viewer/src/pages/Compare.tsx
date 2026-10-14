import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { Download, Upload } from 'lucide-react';
import { apiClient, type JobResponse } from '@/api/client';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ProvenanceAttribution } from '@/components/ProvenanceAttribution';
import { downloadJSON } from '@/lib/datasets';
import { parseApiError } from '@/lib/errorHandling';
import { parseElanFile } from '@/lib/parsers/elan';
import { parseVlmAnnotations } from '@/lib/parsers/vlm';
import { provenanceFromJSON } from '@/lib/provenance';
import {
  comparisonCsv,
  differentVideos,
  matchesTruth,
  pairRuns,
  summarize,
  withTruth,
  type PairedMoment,
  type Run,
} from '@/lib/vlmCompare';
import type { ElanTierAnnotation } from '@/types/annotations';

async function loadRun(jobId: string): Promise<Run> {
  const [job, text] = await Promise.all([apiClient.getJob(jobId), apiClient.getResultFileText(jobId, 'vlm_annotation')]);
  const samples = await parseVlmAnnotations(new File([text], 'vlm_annotation.json'));
  return { job, samples, provenance: provenanceFromJSON(JSON.parse(text)) };
}

const videoOf = (job: JobResponse) => job as JobResponse & { video_filename?: string; video_size_bytes?: number };

const promptOf = (run: Run) => {
  const text = run.samples[0]?.prompt ?? '';
  return (text.trim().split('\n')[0] || 'no prompt recorded').slice(0, 70);
};

const PALETTE = ['#2563eb', '#16a34a', '#d97706', '#9333ea', '#0891b2', '#db2777', '#65a30d'];
const colourOf = (label: string) => {
  if (label.startsWith('ERROR')) return '#71717a';
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
};

const STATUS_COLOUR: Record<PairedMoment['status'], string> = {
  agree: 'bg-green-500',
  disagree: 'bg-red-500',
  unpaired: 'bg-zinc-400',
  error: 'bg-amber-500',
};

/**
 * Two VLM runs of the same video, side by side (spec 021): where they
 * disagree, and how each does against ELAN ground truth when it's loaded.
 */
const Compare = () => {
  const [params] = useSearchParams();
  const a = params.get('a') ?? '';
  const b = params.get('b') ?? '';
  const [ignoreCase, setIgnoreCase] = useState(false);
  const [truth, setTruth] = useState<{ name: string; tiers: ElanTierAnnotation[] } | null>(null);
  const [onlyOneRight, setOnlyOneRight] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const truthInput = useRef<HTMLInputElement>(null);

  const runA = useQuery({ queryKey: ['compare-run', a], queryFn: () => loadRun(a), enabled: !!a, retry: false });
  const runB = useQuery({ queryKey: ['compare-run', b], queryFn: () => loadRun(b), enabled: !!b, retry: false });
  const mismatch = runA.data && runB.data ? differentVideos(runA.data, runB.data) : null;

  useEffect(() => {
    if (!a || mismatch || !runA.data) return;
    let url: string | null = null;
    apiClient.getJobVideoUrl(a).then((u) => {
      url = u;
      setVideoUrl(u);
    }, () => setVideoUrl(null));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [a, mismatch, runA.data]);

  const pairs = useMemo(() => {
    if (!runA.data || !runB.data) return [];
    const intervalOf = (run: Run) =>
      run.provenance.kind === 'recorded' ? Number(run.provenance.record.settings?.frame_interval_sec) || undefined : undefined;
    const paired = pairRuns(runA.data.samples, runB.data.samples, {
      ignoreCase,
      intervalA: intervalOf(runA.data),
      intervalB: intervalOf(runB.data),
    });
    return truth ? withTruth(paired, truth.tiers) : paired;
  }, [runA.data, runB.data, ignoreCase, truth]);

  if (!a || !b) {
    return (
      <div className="container mx-auto p-6 max-w-3xl">
        <p>
          Choose two VLM jobs to compare from a job's page (“Compare with…”).{' '}
          <Link className="underline" to="/jobs">
            Jobs
          </Link>
        </p>
      </div>
    );
  }
  const error = runA.error || runB.error;
  if (error) {
    return (
      <div className="container mx-auto p-6 max-w-3xl">
        <Alert variant="destructive">
          <AlertDescription>Couldn't load the VLM results: {parseApiError(error).message}</AlertDescription>
        </Alert>
      </div>
    );
  }
  if (!runA.data || !runB.data) return <p className="p-8 text-sm text-muted-foreground">Loading both runs…</p>;
  if (mismatch) {
    return (
      <div className="container mx-auto p-6 max-w-3xl">
        <Alert variant="destructive">
          <AlertDescription>{mismatch}</AlertDescription>
        </Alert>
      </div>
    );
  }

  const runs = { a: runA.data, b: runB.data };
  const summary = summarize(pairs, runs.a.samples.length, runs.b.samples.length);
  const duration = Math.max(...pairs.map((p) => p.time), 1);
  const listed = pairs
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.status === 'disagree' && (!onlyOneRight || matchesTruth(p.a, p.truth) !== matchesTruth(p.b, p.truth)));
  const current = selected !== null ? pairs[selected] : null;

  const select = (i: number) => {
    setSelected(i);
    if (videoRef.current) videoRef.current.currentTime = pairs[i].time;
  };

  const loadTruth = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) setTruth({ name: file.name, tiers: await parseElanFile(file) });
  };

  const row = (title: React.ReactNode, cell: (p: PairedMoment) => { colour: string; text: string } | null) => (
    <div className="flex items-center gap-2">
      <div className="w-56 shrink-0 text-xs">{title}</div>
      <div className="relative h-6 flex-1 rounded bg-muted">
        {pairs.map((p, i) => {
          const c = cell(p);
          return c ? (
            <button
              key={i}
              type="button"
              title={`${p.time}s: ${c.text}`}
              onClick={() => select(i)}
              className={`absolute top-0 h-6 w-2 -ml-1 rounded-sm ${selected === i ? 'ring-2 ring-foreground' : ''}`}
              style={{ left: `${(p.time / duration) * 100}%`, background: c.colour }}
            />
          ) : null;
        })}
      </div>
    </div>
  );

  const runTitle = (run: Run, letter: string) => (
    <div>
      <Link className="font-medium underline" to={`/jobs/${run.job.id}`}>
        {letter}: {videoOf(run.job).video_filename ?? run.job.id.slice(0, 8)} · {run.samples[0]?.model ?? '?'}
      </Link>
      <div className="text-muted-foreground truncate" title={run.samples[0]?.prompt}>
        {promptOf(run)}
      </div>
      <ProvenanceAttribution info={run.provenance} pipeline="vlm_annotation" />
    </div>
  );

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Compare two VLM runs</h1>
          <p className="text-muted-foreground mt-1">
            Same video, two runs. Labels are compared as recorded{ignoreCase ? ', ignoring case' : ''}.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={ignoreCase} onChange={(e) => setIgnoreCase(e.target.checked)} /> Ignore case
          </label>
          <input ref={truthInput} type="file" accept=".eaf" className="hidden" onChange={loadTruth} />
          <Button variant="outline" size="sm" onClick={() => truthInput.current?.click()}>
            <Upload className="h-4 w-4 mr-1" /> {truth ? `Ground truth: ${truth.name}` : 'Add ELAN ground truth'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const blob = new Blob([comparisonCsv(pairs, a.slice(0, 8), b.slice(0, 8))], { type: 'text/csv' });
              const link = document.createElement('a');
              link.href = URL.createObjectURL(blob);
              link.download = `compare_${a.slice(0, 8)}_${b.slice(0, 8)}.csv`;
              link.click();
              setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
            }}
          >
            <Download className="h-4 w-4 mr-1" /> CSV
          </Button>
          <Button variant="ghost" size="sm" onClick={() => downloadJSON(`compare_${a.slice(0, 8)}_${b.slice(0, 8)}.json`, { summary, pairs })}>
            JSON
          </Button>
        </div>
      </div>

      <Card className="p-4 space-y-2">
        {row(runTitle(runs.a, 'A'), (p) => (p.a ? { colour: colourOf(p.a.label), text: p.a.label } : null))}
        {row(runTitle(runs.b, 'B'), (p) => (p.b ? { colour: colourOf(p.b.label), text: p.b.label } : null))}
        {truth && row(<span className="font-medium">Ground truth · {truth.name}</span>, (p) => ({ colour: p.truth === 'NO_TOUCH' ? '#a1a1aa' : '#0f766e', text: p.truth ?? '' }))}
        {row(<span className="text-muted-foreground">Agreement</span>, (p) => ({
          colour: { agree: '#22c55e', disagree: '#ef4444', unpaired: '#a1a1aa', error: '#f59e0b' }[p.status],
          text: p.status,
        }))}
        <div className="flex flex-wrap gap-3 pl-[14.5rem] text-xs text-muted-foreground">
          {(['agree', 'disagree', 'unpaired', 'error'] as const).map((st) => (
            <span key={st} className="flex items-center gap-1">
              <span className={`inline-block h-2 w-2 rounded-sm ${STATUS_COLOUR[st]}`} /> {st}
            </span>
          ))}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="p-4 space-y-2 text-sm">
          <h3 className="font-medium">Summary</h3>
          <p>
            {summary.compared} moments compared: <strong>{summary.agree}</strong> agree,{' '}
            <strong>{summary.disagree}</strong> disagree
            {summary.agreementRate !== null && ` (${Math.round(summary.agreementRate * 100)}% agreement)`}.
          </p>
          <p className="text-muted-foreground">
            A has {summary.samplesA} samples, B has {summary.samplesB}; {summary.unpaired} unpaired, {summary.errors} with errors.
          </p>
          {summary.truth && (
            <p>
              Against ground truth ({summary.truth.compared} moments): A agrees on {summary.truth.agreeA}, B on {summary.truth.agreeB}.
            </p>
          )}
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th>A said</th>
                <th>B said</th>
                <th className="text-right">times</th>
              </tr>
            </thead>
            <tbody>
              {summary.labelPairs.map((lp) => (
                <tr key={`${lp.a}-${lp.b}`} className={lp.a === lp.b ? '' : 'text-red-600 dark:text-red-400'}>
                  <td>{lp.a}</td>
                  <td>{lp.b}</td>
                  <td className="text-right">{lp.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card className="p-4 space-y-2 text-sm">
          <div className="flex items-center justify-between">
            <h3 className="font-medium">Where they disagree ({listed.length})</h3>
            {truth && (
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" checked={onlyOneRight} onChange={(e) => setOnlyOneRight(e.target.checked)} /> only one matches truth
              </label>
            )}
          </div>
          <ul className="max-h-80 overflow-y-auto divide-y">
            {listed.map(({ p, i }) => (
              <li key={i}>
                <button type="button" className={`w-full text-left py-1 ${selected === i ? 'font-medium' : ''}`} onClick={() => select(i)}>
                  {p.time}s: A {p.a?.label} · B {p.b?.label}
                  {p.truth ? ` · truth ${p.truth}` : ''}
                </button>
              </li>
            ))}
            {listed.length === 0 && <li className="text-muted-foreground">None.</li>}
          </ul>
        </Card>

        <Card className="p-4 space-y-2 text-sm">
          <h3 className="font-medium">{current ? `At ${current.time}s` : 'Select a moment'}</h3>
          {videoUrl ? (
            <video ref={videoRef} src={videoUrl} controls className="w-full rounded" />
          ) : (
            <p className="text-xs text-muted-foreground">Loading the video…</p>
          )}
          {current &&
            (['a', 'b'] as const).map((key) => {
              const sample = current[key];
              return (
                <div key={key} className="space-y-1">
                  <div>
                    {key.toUpperCase()}: {sample ? <Badge style={{ background: colourOf(sample.label) }}>{sample.label}</Badge> : 'no sample here'}
                  </div>
                  {sample && (
                    <details className="text-xs">
                      <summary className="cursor-pointer text-muted-foreground">Reasoning and raw response</summary>
                      <p className="whitespace-pre-wrap">{sample.reasoning}</p>
                      <pre className="whitespace-pre-wrap text-muted-foreground">{sample.raw_response}</pre>
                    </details>
                  )}
                </div>
              );
            })}
        </Card>
      </div>
    </div>
  );
};

export default Compare;
