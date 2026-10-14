import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { AlertTriangle, Loader2, Play, Plus, Trash2, X } from 'lucide-react';
import { apiClient, type JobResponse } from '@/api/client';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { parseApiError } from '@/lib/errorHandling';
import { wizardState } from '@/lib/wizardStart';
import type { VlmPreviewResponse } from '@/types/pipelines';
import type { WorkbenchStart } from '@/pages/Prompts';
import { evenlySpaced, parseMoments } from '@/lib/workbench';

type Sampling = 'single_frame' | 'frame_burst';

interface Cell {
  moment: number;
  prompt: string;
  model: string;
  result?: VlmPreviewResponse;
  error?: string;
}

interface Round {
  id: number;
  videoLabel: string;
  sampling: Sampling;
  cells: Cell[];
}

const BURST_OFFSETS = [-2, -1, 0, 1, 2];
const LOCAL_HOSTS = ['localhost', '127.0.0.1', '::1', '[::1]', 'host.docker.internal'];

const shortPrompt = (p: string) => (p.trim().split('\n')[0] || '(empty prompt)').slice(0, 60);
const labelOfJob = (job: JobResponse) =>
  ((job as JobResponse & Record<string, unknown>).video_filename as string | undefined) ?? `job ${job.id.slice(0, 8)}`;

/**
 * The prompt workbench (spec 020): try prompts × models × moments of a video
 * side by side, keep earlier rounds to compare, send the winner to a job.
 * Every prompt run here is kept in the prompt library.
 */
const Workbench = () => {
  const navigate = useNavigate();
  const location = useLocation();
  // From the Prompts page (router state) or the wizard's test panel (URL, new tab).
  const query = new URLSearchParams(location.search);
  const start = (location.state as Partial<WorkbenchStart> | null)?.workbench ?? {
    prompts: query.get('prompt') ? [query.get('prompt') as string] : undefined,
    model: query.get('model') ?? undefined,
  };
  const [videoPath, setVideoPath] = useState(start?.videoPath ?? '');
  const [videoLabel, setVideoLabel] = useState(start?.videoPath ?? '');
  const [duration, setDuration] = useState(0);
  const [momentsText, setMomentsText] = useState('');
  const [sampling, setSampling] = useState<Sampling>('single_frame');
  const [interval, setInterval] = useState(1);
  const [prompts, setPrompts] = useState<string[]>(start?.prompts?.length ? start.prompts : ['']);
  const [models, setModels] = useState<string[]>(start?.model ? [start.model] : []);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const { data: jobsData } = useQuery({ queryKey: ['jobs', 'workbench'], queryFn: () => apiClient.getJobs(1, 50), retry: false });
  const { data: modelData, error: modelError } = useQuery({ queryKey: ['vlm-models'], queryFn: () => apiClient.getVlmModels(), retry: false });
  const { data: libraryData } = useQuery({ queryKey: ['prompts', 'workbench'], queryFn: () => apiClient.listPrompts(), retry: false });

  const jobs = (jobsData?.jobs ?? []).filter((j) => j.video_path);
  const moments = parseMoments(momentsText);
  const combos = prompts.filter((p) => p.trim()).length * models.length * moments.length;
  const baseUrl = modelData?.baseUrl ?? '';
  const remoteServer = useMemo(() => {
    try {
      return baseUrl !== '' && !LOCAL_HOSTS.includes(new URL(baseUrl).hostname);
    } catch {
      return false;
    }
  }, [baseUrl]);

  const chooseJob = (id: string) => {
    const job = jobs.find((j) => j.id === id);
    if (!job) return;
    setVideoPath(job.video_path ?? '');
    setVideoLabel(labelOfJob(job));
    setDuration(Number((job as JobResponse & Record<string, unknown>).video_duration_seconds) || 0);
  };

  const run = async () => {
    const usable = prompts.filter((p) => p.trim());
    const round: Round = { id: Date.now(), videoLabel: videoLabel || videoPath, sampling, cells: [] };
    for (const moment of moments) for (const prompt of usable) for (const model of models) round.cells.push({ moment, prompt, model });
    setRounds((prev) => [round, ...prev]);
    setProgress({ done: 0, total: round.cells.length });
    // One at a time: a local model server serves one request well, not twelve.
    for (const [i, cell] of round.cells.entries()) {
      try {
        cell.result = await apiClient.previewVlmPrompt({
          videoPath,
          timestampSec: cell.moment,
          prompt: cell.prompt,
          model: cell.model,
          samplingMode: sampling,
          frameIntervalSec: interval,
          burstOffsets: sampling === 'frame_burst' ? BURST_OFFSETS : undefined,
        });
      } catch (e) {
        cell.error = parseApiError(e).message;
      }
      setRounds((prev) => prev.map((r) => (r.id === round.id ? { ...round, cells: [...round.cells] } : r)));
      setProgress({ done: i + 1, total: round.cells.length });
    }
    setProgress(null);
  };

  const sendToJob = (cell: Cell, roundSampling: Sampling) =>
    navigate('/jobs/new', {
      state: wizardState({
        mode: 'settings',
        label: `the workbench (${cell.model})`,
        selectedPipelines: ['vlm_annotation'],
        config: {
          vlm_annotation: {
            prompt: cell.prompt,
            model: cell.model,
            sampling_mode: roundSampling,
            ...(roundSampling === 'frame_burst' ? { burst_offsets: BURST_OFFSETS, frame_interval_sec: interval } : {}),
          },
        },
      }),
    });

  const saveAsPreset = async (cell: Cell, roundSampling: Sampling) => {
    try {
      const preset = await apiClient.createPreset({
        name: `${shortPrompt(cell.prompt)} (${cell.model})`,
        selected_pipelines: ['vlm_annotation'],
        config: { vlm_annotation: { prompt: cell.prompt, model: cell.model, sampling_mode: roundSampling } },
      });
      setNotice(`Saved preset “${preset.name}”.`);
    } catch (e) {
      setNotice(`Couldn't save the preset: ${parseApiError(e).message}`);
    }
  };

  const columns = (round: Round) => [...new Map(round.cells.map((c) => [`${c.prompt}\u0000${c.model}`, c])).values()];

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Prompt workbench</h1>
          <p className="text-muted-foreground mt-2">
            Try prompts and models on moments of a video, side by side. Earlier rounds stay below to compare. Every
            prompt run here is kept in the <Link className="underline" to="/prompts">prompt library</Link>.
          </p>
        </div>
      </div>

      {remoteServer && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            The model server is at {baseUrl}, not this machine: the frames you test are sent there.
          </AlertDescription>
        </Alert>
      )}
      {modelError && (
        <Alert variant="destructive">
          <AlertDescription>Can't reach the model server: {parseApiError(modelError).message}</AlertDescription>
        </Alert>
      )}
      {notice && (
        <Alert>
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}

      <Card className="p-4 grid gap-4 lg:grid-cols-2">
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="wb-job">Video</Label>
            <select
              id="wb-job"
              className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
              value={jobs.find((j) => j.video_path === videoPath)?.id ?? ''}
              onChange={(e) => chooseJob(e.target.value)}
            >
              <option value="">A video from a past job…</option>
              {jobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {labelOfJob(job)} ({job.id.slice(0, 8)})
                </option>
              ))}
            </select>
            <Input
              aria-label="Video path on the server"
              placeholder="…or a path on the server, e.g. /data/session1/p01.mp4"
              value={videoPath}
              onChange={(e) => {
                setVideoPath(e.target.value);
                setVideoLabel(e.target.value);
                setDuration(0);
              }}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="wb-moments">Moments (seconds)</Label>
            <div className="flex gap-2">
              <Input id="wb-moments" placeholder="e.g. 2, 5.5, 9" value={momentsText} onChange={(e) => setMomentsText(e.target.value)} />
              {duration > 0 && (
                <Button variant="outline" onClick={() => setMomentsText(evenlySpaced(5, duration).join(', '))}>
                  5 across the video
                </Button>
              )}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-1">
              <input type="radio" checked={sampling === 'single_frame'} onChange={() => setSampling('single_frame')} /> Single frame
            </label>
            <label className="flex items-center gap-1">
              <input type="radio" checked={sampling === 'frame_burst'} onChange={() => setSampling('frame_burst')} /> Burst of 5
            </label>
            {sampling === 'frame_burst' && (
              <label className="flex items-center gap-1">
                every
                <Input className="h-8 w-20" type="number" min={0.1} step={0.1} value={interval} onChange={(e) => setInterval(Number(e.target.value) || 1)} />s
              </label>
            )}
          </div>
          <div className="space-y-1">
            <Label>Models</Label>
            <div className="flex flex-wrap gap-3 text-sm">
              {(modelData?.models ?? []).map((m) => (
                <label key={m} className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={models.includes(m)}
                    onChange={() => setModels((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]))}
                  />
                  {m}
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Prompts</Label>
            <select
              aria-label="Load a prompt from the library"
              className="h-8 rounded-md border border-input bg-background px-2 text-xs"
              value=""
              onChange={(e) => {
                const p = libraryData?.prompts.find((x) => x.sha256 === e.target.value);
                if (p) setPrompts((prev) => [...prev.filter((x) => x.trim()), p.text]);
              }}
            >
              <option value="">Load from the library…</option>
              {(libraryData?.prompts ?? []).map((p) => (
                <option key={p.sha256} value={p.sha256}>
                  {p.starred ? '★ ' : ''}
                  {p.name || shortPrompt(p.text)}
                </option>
              ))}
            </select>
          </div>
          {prompts.map((p, i) => (
            <div key={i} className="flex gap-2">
              <Textarea
                aria-label={`Prompt ${i + 1}`}
                rows={4}
                value={p}
                onChange={(e) => setPrompts((prev) => prev.map((x, j) => (j === i ? e.target.value : x)))}
              />
              {prompts.length > 1 && (
                <Button size="icon" variant="ghost" aria-label={`Remove prompt ${i + 1}`} onClick={() => setPrompts((prev) => prev.filter((_, j) => j !== i))}>
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>
          ))}
          <Button size="sm" variant="outline" onClick={() => setPrompts((prev) => [...prev, prev[prev.length - 1] ?? ''])}>
            <Plus className="h-4 w-4 mr-1" /> Another prompt (copy of the last)
          </Button>
        </div>

        <div className="lg:col-span-2 flex items-center gap-3">
          <Button onClick={run} disabled={!videoPath || combos === 0 || progress !== null}>
            {progress ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
            {progress ? `Running ${progress.done}/${progress.total}…` : `Run ${combos} combination${combos === 1 ? '' : 's'}`}
          </Button>
          {rounds.length > 0 && !progress && (
            <Button variant="ghost" onClick={() => setRounds([])}>
              <Trash2 className="h-4 w-4 mr-1" /> Clear rounds
            </Button>
          )}
        </div>
      </Card>

      {rounds.map((round, ri) => (
        <Card key={round.id} className="p-4 space-y-3 overflow-x-auto">
          <h3 className="font-medium">
            Round {rounds.length - ri}: {round.videoLabel} · {round.sampling === 'frame_burst' ? 'burst' : 'single frame'}
          </h3>
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr>
                <th className="text-left p-2 w-20">Moment</th>
                {columns(round).map((c) => (
                  <th key={`${c.prompt}-${c.model}`} className="text-left p-2 align-top font-normal">
                    <div className="font-medium" title={c.prompt}>
                      {shortPrompt(c.prompt)}
                    </div>
                    <Badge variant="outline">{c.model}</Badge>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...new Set(round.cells.map((c) => c.moment))].map((moment) => (
                <tr key={moment} className="border-t align-top">
                  <td className="p-2 font-mono">{moment}s</td>
                  {columns(round).map((col) => {
                    const cell = round.cells.find((c) => c.moment === moment && c.prompt === col.prompt && c.model === col.model)!;
                    return (
                      <td key={`${col.prompt}-${col.model}`} className="p-2 min-w-[220px]">
                        {cell.error ? (
                          <p className="text-destructive text-xs">{cell.error}</p>
                        ) : !cell.result ? (
                          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                        ) : (
                          <div className="space-y-1">
                            <div className="flex gap-1">
                              {cell.result.frames.map((f, k) => (
                                <img
                                  key={k}
                                  alt={`frame ${f.frameNumber ?? ''}`}
                                  title={`frame ${f.frameNumber ?? '?'} at ${f.timestampSec ?? '?'}s`}
                                  className="h-12 rounded"
                                  src={`data:image/jpeg;base64,${f.jpegBase64}`}
                                />
                              ))}
                            </div>
                            <Badge>{cell.result.label}</Badge>
                            <span className="ml-2 text-xs text-muted-foreground">{cell.result.totalTime.toFixed(1)}s</span>
                            <details className="text-xs">
                              <summary className="cursor-pointer text-muted-foreground">Reasoning</summary>
                              <p className="whitespace-pre-wrap">{cell.result.reasoning}</p>
                            </details>
                            <div className="flex gap-2 text-xs">
                              <button type="button" className="underline" onClick={() => sendToJob(cell, round.sampling)}>
                                Send to job
                              </button>
                              <button type="button" className="underline" onClick={() => saveAsPreset(cell, round.sampling)}>
                                Save as preset
                              </button>
                            </div>
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}
    </div>
  );
};

export default Workbench;
