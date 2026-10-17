// The Hugging Face token, set up from the pipeline that needs it (VideoAnnotator
// spec 031): make a token, accept the licence, paste it. The server checks and
// saves it; the viewer never sees it again, only whether it's set.
import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';

import { apiClient } from '@/api/client';
import { APIError, apiErrorEnvelope } from '@/api/handleError';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import type { HuggingFaceTokenStatus, ReadinessItem } from '@/types/pipelines';

const TOKENS_PAGE = 'https://huggingface.co/settings/tokens';

type Outcome =
  | { kind: 'saved'; status: HuggingFaceTokenStatus }
  | { kind: 'rejected' }
  | { kind: 'forbidden' }
  | { kind: 'failed'; message: string };

/** What a save tells the user, in a sentence. */
const outcomeMessage = (outcome: Outcome): string => {
  switch (outcome.kind) {
    case 'saved': {
      const { status } = outcome;
      if (status.source === 'environment') {
        return `Saved, but the server's environment sets ${status.envVar ?? 'a token'}, which is used instead.`;
      }
      if (status.check === 'unknown') return "Token saved. Couldn't reach Hugging Face to check it yet.";
      return `Token saved and works${status.account ? ` (account: ${status.account})` : ''}.`;
    }
    case 'rejected':
      return 'Hugging Face rejected that token (expired, revoked or mis-pasted). Nothing was saved.';
    case 'forbidden':
      return 'Only an admin can add the token.';
    case 'failed':
      return `Couldn't save the token: ${outcome.message}`;
  }
};

interface HuggingFaceTokenSetupProps {
  /** The pipeline's licence items (blockers: not accepted; notes: not checked yet). */
  licences: ReadinessItem[];
  /** Model pages whose licence is known not to be accepted. */
  notAccepted: string[];
  tokenHelpUrl?: string | null;
  isAdmin: boolean | 'unknown';
  /** Re-check readiness, after a save or after accepting a licence. */
  onCheckAgain?: () => void;
  isChecking?: boolean;
}

export const HuggingFaceTokenSetup = ({
  licences,
  notAccepted,
  tokenHelpUrl,
  isAdmin,
  onCheckAgain,
  isChecking
}: HuggingFaceTokenSetupProps) => {
  const [token, setToken] = useState('');
  const [saving, setSaving] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  const save = async () => {
    setSaving(true);
    try {
      const status = await apiClient.saveHuggingFaceToken(token.trim());
      setOutcome({ kind: 'saved', status });
      setToken('');
      onCheckAgain?.();
    } catch (error) {
      if (apiErrorEnvelope(error).code === 'HF_TOKEN_REJECTED') setOutcome({ kind: 'rejected' });
      else if (error instanceof APIError && error.status === 403) setOutcome({ kind: 'forbidden' });
      else setOutcome({ kind: 'failed', message: error instanceof Error ? error.message : String(error) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mt-1 space-y-2 rounded border border-amber-500/40 bg-amber-500/5 p-2 text-xs">
      <p className="font-medium text-foreground">
        This pipeline needs a free Hugging Face token, to download its models.
      </p>
      <ol className="list-decimal space-y-1.5 pl-4 text-muted-foreground">
        <li>
          Create a Hugging Face account and a <span className="font-medium">Read</span> token.{' '}
          <a href={tokenHelpUrl ?? TOKENS_PAGE} target="_blank" rel="noreferrer" className="underline">
            Open huggingface.co
          </a>
        </li>
        {licences.length > 0 && (
          <li>
            Accept {licences.length === 1 ? "the model's licence" : "each model's licence"}, signed in as that account:
            <ul className="mt-0.5 space-y-0.5">
              {licences.map((l) => {
                const refused = notAccepted.includes(l.name);
                return (
                  <li key={l.name} className="flex items-center gap-1">
                    {refused ? (
                      <XCircle className="h-3.5 w-3.5 text-amber-600" aria-label="not accepted yet" />
                    ) : (
                      <Circle className="h-3.5 w-3.5" aria-label="not checked yet" />
                    )}
                    {l.helpUrl ? (
                      <a href={l.helpUrl} target="_blank" rel="noreferrer" className="underline">
                        {l.name}
                      </a>
                    ) : (
                      l.name
                    )}
                    {refused && <span className="text-amber-700 dark:text-amber-400">not accepted yet</span>}
                  </li>
                );
              })}
            </ul>
          </li>
        )}
        <li>
          {isAdmin === false ? (
            <>Ask whoever runs this VideoAnnotator to add a Hugging Face token (here or in Settings).</>
          ) : (
            <form
              className="flex flex-wrap items-center gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                if (token.trim()) void save();
              }}
            >
              <label htmlFor="hf-token" className="text-muted-foreground">
                Paste the token here
              </label>
              <Input
                id="hf-token"
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder="hf_…"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                className="h-7 w-56 text-xs"
              />
              <Button type="submit" size="sm" className="h-7 px-2 text-[11px]" disabled={!token.trim() || saving}>
                {saving && <Loader2 className="mr-1 h-3 w-3 animate-spin" aria-hidden="true" />}
                Save and check
              </Button>
            </form>
          )}
        </li>
      </ol>
      {outcome && (
        <p
          role="status"
          className={`flex items-center gap-1 ${outcome.kind === 'saved' ? 'text-green-700 dark:text-green-400' : 'text-amber-700 dark:text-amber-400'}`}
        >
          {outcome.kind === 'saved' && <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />}
          {outcomeMessage(outcome)}
        </p>
      )}
      {onCheckAgain && (
        <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={onCheckAgain} disabled={isChecking}>
          {isChecking ? <Loader2 className="mr-1 h-3 w-3 animate-spin" aria-hidden="true" /> : null}
          Check again
        </Button>
      )}
    </div>
  );
};

/** Settings: whether the server has a token, where from, and change or remove it (spec 031). */
export const HuggingFaceTokenCard = ({ isAdmin }: { isAdmin: boolean | 'unknown' }) => {
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: ['videoannotator', 'huggingface-token'],
    queryFn: () => apiClient.getHuggingFaceToken(),
    retry: false,
  });
  const [editing, setEditing] = useState(false);
  const [token, setToken] = useState('');
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const refresh = (next: HuggingFaceTokenStatus) => {
    queryClient.setQueryData(['videoannotator', 'huggingface-token'], next);
    // Readiness depends on the token; the client caches the catalog itself too.
    apiClient.clearPipelineCache();
    void queryClient.invalidateQueries({ queryKey: ['videoannotator', 'pipelines'] });
  };
  const save = useMutation({
    mutationFn: (value: string) => apiClient.saveHuggingFaceToken(value),
    onSuccess: (next) => {
      refresh(next);
      setOutcome({ kind: 'saved', status: next });
      setToken('');
      setEditing(false);
    },
    onError: (error) => {
      if (apiErrorEnvelope(error).code === 'HF_TOKEN_REJECTED') setOutcome({ kind: 'rejected' });
      else if (error instanceof APIError && error.status === 403) setOutcome({ kind: 'forbidden' });
      else setOutcome({ kind: 'failed', message: error instanceof Error ? error.message : String(error) });
    },
  });
  const remove = useMutation({
    mutationFn: () => apiClient.removeHuggingFaceToken(),
    onSuccess: (next) => {
      refresh(next);
      setOutcome(null);
    },
  });

  const s = status.data;
  const describe = (): string => {
    if (!s) return status.isError ? "Couldn't ask the server." : 'Checking…';
    if (!s.set) return 'Not set. Speaker turns needs one to download its models.';
    const where = s.source === 'environment' ? `from ${s.envVar} in the server's environment` : 'saved on this server';
    const account = s.account ? `account ${s.account}, ` : '';
    const check = s.check === 'invalid' ? ' Hugging Face rejects it.' : s.check === 'unknown' ? " Couldn't check it with Hugging Face." : '';
    return `Set (${account}${where}).${check}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Hugging Face token</CardTitle>
        <CardDescription>
          Lets the server download models whose licence you've accepted on Hugging Face. It stays on this server.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <p>{describe()}</p>
        {s?.source === 'environment' && (
          <p className="text-xs text-muted-foreground">
            The environment's token is the one used; one saved here wouldn't replace it.
          </p>
        )}
        {isAdmin !== false && s && s.source !== 'environment' && !editing && (
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={() => setEditing(true)}>
              {s.set ? 'Change' : 'Add a token'}
            </Button>
            {s.source === 'saved' && (
              <Button size="sm" variant="outline" onClick={() => remove.mutate()} disabled={remove.isPending}>
                Remove
              </Button>
            )}
          </div>
        )}
        {editing && (
          <form
            className="flex flex-wrap items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              if (token.trim()) save.mutate(token.trim());
            }}
          >
            <label htmlFor="hf-token-settings" className="sr-only">
              Hugging Face token
            </label>
            <Input
              id="hf-token-settings"
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder="hf_…"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              className="h-8 w-64"
            />
            <Button type="submit" size="sm" disabled={!token.trim() || save.isPending}>
              {save.isPending && <Loader2 className="mr-1 h-3 w-3 animate-spin" aria-hidden="true" />}
              Save and check
            </Button>
            <Button type="button" size="sm" variant="ghost" onClick={() => setEditing(false)}>
              Cancel
            </Button>
            <a href={TOKENS_PAGE} target="_blank" rel="noreferrer" className="text-xs underline">
              Get a Read token
            </a>
          </form>
        )}
        {outcome && (
          <p role="status" className="text-xs text-muted-foreground">
            {outcomeMessage(outcome)}
          </p>
        )}
      </CardContent>
    </Card>
  );
};
