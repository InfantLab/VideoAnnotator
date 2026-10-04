import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { EyeOff, FlaskConical, Play, Search, Star, Trash2 } from 'lucide-react';
import { apiClient } from '@/api/client';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { parseApiError } from '@/lib/errorHandling';
import { whitespaceOnly, wordDiff } from '@/lib/wordDiff';
import { wizardState } from '@/lib/wizardStart';
import type { LibraryPrompt, PromptUpdateRequest } from '@/types/prompts';

const when = (iso: string) => new Date(iso).toLocaleString();
const titleOf = (p: LibraryPrompt) => p.name || p.text.trim().split('\n')[0].slice(0, 90);

/** Where a workbench session starts from (prompts to try). */
export interface WorkbenchStart {
  workbench: { prompts?: string[]; model?: string; videoPath?: string };
}

const Diff = ({ a, b }: { a: LibraryPrompt; b: LibraryPrompt }) => {
  if (whitespaceOnly(a.text, b.text)) {
    return <p className="text-sm">These two differ only in spaces or line breaks.</p>;
  }
  return (
    <pre className="whitespace-pre-wrap text-sm font-mono rounded-md bg-muted p-3">
      {wordDiff(a.text, b.text).map((part, i) =>
        part.kind === 'same' ? (
          <span key={i}>{part.text}</span>
        ) : part.kind === 'removed' ? (
          <del key={i} className="bg-red-200 text-red-900 dark:bg-red-950 dark:text-red-200">
            {part.text}
          </del>
        ) : (
          <ins key={i} className="bg-green-200 text-green-900 no-underline dark:bg-green-950 dark:text-green-200">
            {part.text}
          </ins>
        ),
      )}
    </pre>
  );
};

const PromptDetail = ({ prompt }: { prompt: LibraryPrompt }) => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [name, setName] = useState(prompt.name ?? '');
  const [tags, setTags] = useState(prompt.tags.join(', '));
  const [problem, setProblem] = useState<string | null>(null);
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['prompts'] });
  const update = useMutation({
    mutationFn: (body: PromptUpdateRequest) => apiClient.updatePrompt(prompt.sha256, body),
    onSuccess: refresh,
    onError: (e) => setProblem(parseApiError(e).message),
  });
  const remove = useMutation({
    mutationFn: () => apiClient.deletePrompt(prompt.sha256),
    onSuccess: refresh,
    onError: (e) => setProblem(parseApiError(e).message),
  });

  const useInJob = () =>
    navigate('/jobs/new', {
      state: wizardState({
        mode: 'settings',
        label: `prompt “${titleOf(prompt)}”`,
        selectedPipelines: ['vlm_annotation'],
        config: { vlm_annotation: { prompt: prompt.text, ...(prompt.models[0] ? { model: prompt.models[0] } : {}) } },
      }),
    });

  return (
    <div className="space-y-3 border-t p-4">
      <pre className="whitespace-pre-wrap text-sm font-mono rounded-md bg-muted p-3">{prompt.text}</pre>
      <p className="text-xs text-muted-foreground font-mono break-all">sha256 {prompt.sha256}</p>
      <p className="text-xs text-muted-foreground">
        First used {when(prompt.first_used_at)}, last used {when(prompt.last_used_at)} · {prompt.use_count} use
        {prompt.use_count === 1 ? '' : 's'} with {prompt.models.join(', ') || 'no model recorded'}
        {prompt.updated_by ? ` · last edited by ${prompt.updated_by}` : ''}
      </p>
      {prompt.job_ids.length > 0 && (
        <p className="text-xs">
          Jobs:{' '}
          {prompt.job_ids.map((id, i) => (
            <span key={id}>
              {i > 0 && ', '}
              <Link className="underline" to={`/jobs/${id}`}>
                {id.slice(0, 8)}
              </Link>
            </span>
          ))}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Input className="h-8 w-56" placeholder="Name" aria-label="Prompt name" value={name} onChange={(e) => setName(e.target.value)} />
        <Input className="h-8 w-56" placeholder="Tags, comma-separated" aria-label="Tags" value={tags} onChange={(e) => setTags(e.target.value)} />
        <Button size="sm" variant="outline" onClick={() => update.mutate({ name, tags: tags.split(',') })}>
          Save
        </Button>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={useInJob}>
          <Play className="h-4 w-4 mr-1" /> Use in a new job
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() =>
            navigate('/workbench', {
              state: { workbench: { prompts: [prompt.text], model: prompt.models[0] } } satisfies WorkbenchStart,
            })
          }
        >
          <FlaskConical className="h-4 w-4 mr-1" /> Open in workbench
        </Button>
        <Button size="sm" variant="ghost" onClick={() => update.mutate({ hidden: !prompt.hidden })}>
          <EyeOff className="h-4 w-4 mr-1" /> {prompt.hidden ? 'Show in the list' : 'Hide'}
        </Button>
        {prompt.job_ids.length === 0 && (
          <Button size="sm" variant="ghost" onClick={() => remove.mutate()}>
            <Trash2 className="h-4 w-4 mr-1" /> Delete
          </Button>
        )}
      </div>
      {problem && <p className="text-sm text-destructive">{problem}</p>}
    </div>
  );
};

/**
 * Every VLM prompt that ran on this server, from jobs and previews (spec 020),
 * kept once per exact text.
 */
const Prompts = () => {
  const queryClient = useQueryClient();
  const [q, setQ] = useState('');
  const [includeHidden, setIncludeHidden] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [compare, setCompare] = useState<string[]>([]);
  const { data, isLoading, error } = useQuery({
    queryKey: ['prompts', q, includeHidden],
    queryFn: () => apiClient.listPrompts({ q: q.trim() || undefined, includeHidden }),
  });
  const star = useMutation({
    mutationFn: (p: LibraryPrompt) => apiClient.updatePrompt(p.sha256, { starred: !p.starred }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['prompts'] }),
  });
  const prompts = data?.prompts ?? [];
  const toggleCompare = (sha: string) =>
    setCompare((prev) => (prev.includes(sha) ? prev.filter((s) => s !== sha) : [...prev, sha].slice(-2)));
  const compared = compare.map((sha) => prompts.find((p) => p.sha256 === sha)).filter(Boolean) as LibraryPrompt[];

  return (
    <div className="container mx-auto p-6 max-w-5xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Prompts</h1>
          <p className="text-muted-foreground mt-2">
            Every VLM prompt run on this server, in a job or a preview, kept once per exact text. Name, star and
            compare them; reuse one in a job or try it in the workbench.
          </p>
        </div>
        <Link to="/workbench">
          <Button variant="outline">
            <FlaskConical className="h-4 w-4 mr-2" /> Workbench
          </Button>
        </Link>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input className="pl-8 w-80" placeholder="Search text or name" aria-label="Search prompts" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={includeHidden} onChange={(e) => setIncludeHidden(e.target.checked)} />
          Show hidden
        </label>
        <span className="text-xs text-muted-foreground">Tick two to compare them.</span>
      </div>

      {compared.length === 2 && (
        <Card className="p-4 space-y-2">
          <p className="text-sm font-medium">
            <del className="no-underline bg-red-200 dark:bg-red-950 px-1">{titleOf(compared[0])}</del> →{' '}
            <ins className="no-underline bg-green-200 dark:bg-green-950 px-1">{titleOf(compared[1])}</ins>
          </p>
          <Diff a={compared[0]} b={compared[1]} />
        </Card>
      )}

      <Card>
        {isLoading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading…</p>
        ) : error ? (
          <Alert variant="destructive" className="m-4">
            <AlertDescription>Couldn't load prompts: {parseApiError(error).message}</AlertDescription>
          </Alert>
        ) : prompts.length === 0 ? (
          <p className="p-8 text-center text-sm text-muted-foreground">
            {q ? 'No prompt matches.' : 'No prompts yet. They are added here when a VLM job or a preview runs one.'}
          </p>
        ) : (
          <ul className="divide-y">
            {prompts.map((p) => (
              <li key={p.sha256} className={p.hidden ? 'opacity-60' : ''}>
                <div className="flex items-center gap-3 p-3">
                  <input
                    type="checkbox"
                    aria-label={`Compare ${titleOf(p)}`}
                    checked={compare.includes(p.sha256)}
                    onChange={() => toggleCompare(p.sha256)}
                  />
                  <button type="button" aria-label={p.starred ? 'Unstar' : 'Star'} onClick={() => star.mutate(p)}>
                    <Star className={`h-4 w-4 ${p.starred ? 'fill-yellow-400 text-yellow-500' : 'text-muted-foreground'}`} />
                  </button>
                  <button type="button" className="min-w-0 flex-1 text-left" onClick={() => setOpen(open === p.sha256 ? null : p.sha256)}>
                    <div className="truncate font-medium">{titleOf(p)}</div>
                    <div className="text-xs text-muted-foreground">
                      {p.use_count} use{p.use_count === 1 ? '' : 's'} · last {when(p.last_used_at)}
                      {p.job_ids.length > 0 ? ` · ${p.job_ids.length} job${p.job_ids.length === 1 ? '' : 's'}` : ''}
                    </div>
                  </button>
                  <div className="hidden sm:flex flex-wrap gap-1 justify-end">
                    {p.tags.map((t) => (
                      <Badge key={t} variant="secondary">
                        {t}
                      </Badge>
                    ))}
                    {p.models.map((m) => (
                      <Badge key={m} variant="outline">
                        {m}
                      </Badge>
                    ))}
                  </div>
                </div>
                {open === p.sha256 && <PromptDetail prompt={p} />}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
};

export default Prompts;
