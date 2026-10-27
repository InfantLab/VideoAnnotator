// The Hugging Face token, set up from the pipeline that needs it (VideoAnnotator
// spec 031): make a token, accept the licence, paste it. The server checks and
// saves it; the viewer never sees it again, only whether it's set.
import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';

import { Link } from 'react-router-dom';

import { apiClient, hasConfiguredApiToken } from '@/api/client';
import { APIError, apiErrorEnvelope, isNeedsKey, NEEDS_KEY_MESSAGE } from '@/api/handleError';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { StateBadge, type StateTone } from '@/components/settings/StateBadge';
import type { HuggingFaceTokenStatus, ReadinessItem } from '@/types/pipelines';

const TOKENS_PAGE = 'https://huggingface.co/settings/tokens';

type Outcome =
  | { kind: 'saved'; status: HuggingFaceTokenStatus }
  | { kind: 'rejected' }
  | { kind: 'unauthenticated' }
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
    case 'unauthenticated':
      return "The server didn't take the token because this viewer isn't connected with an API key.";
    case 'forbidden':
      return "The server didn't take the token because this viewer's API key isn't an admin key.";
    case 'failed':
      return `Couldn't save the token: ${outcome.message}`;
  }
};

const outcomeOfError = (error: unknown): Outcome => {
  if (apiErrorEnvelope(error).code === 'HF_TOKEN_REJECTED') return { kind: 'rejected' };
  if (error instanceof APIError && error.status === 401) return { kind: 'unauthenticated' };
  if (error instanceof APIError && error.status === 403) return { kind: 'forbidden' };
  return { kind: 'failed', message: error instanceof Error ? error.message : String(error) };
};

const CLI_COMMAND = 'videoannotator settings huggingface-token';

/** What to do when the viewer can't save the token itself. */
const NextStep = ({ kind }: { kind: 'no-key' | 'unauthenticated' | 'forbidden' }) => (
  <span className="text-muted-foreground">
    {kind === 'forbidden' ? (
      <>Ask whoever runs this VideoAnnotator, or make an admin key with </>
    ) : (
      <>
        Connect this viewer first: open the link the server printed when it first started, or follow{' '}
      </>
    )}
    <Link to="/settings" className="underline">
      the steps in Settings
    </Link>
    . Or, on the server&apos;s computer, run{' '}
    <code className="rounded bg-muted px-1">{CLI_COMMAND}</code>.
  </span>
);

const OutcomeLine = ({ outcome }: { outcome: Outcome }) => (
  <p
    role="status"
    className={`flex flex-wrap items-center gap-1 ${outcome.kind === 'saved' ? 'text-green-700 dark:text-green-400' : 'text-amber-700 dark:text-amber-400'}`}
  >
    {outcome.kind === 'saved' && <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />}
    {outcomeMessage(outcome)}
    {(outcome.kind === 'unauthenticated' || outcome.kind === 'forbidden') && <NextStep kind={outcome.kind} />}
  </p>
);

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
      setOutcome(outcomeOfError(error));
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
          ) : !hasConfiguredApiToken() ? (
            <>
              Paste the token here once this viewer is connected to the server with an admin key.{' '}
              <NextStep kind="no-key" />
            </>
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
      {outcome && <OutcomeLine outcome={outcome} />}
      {onCheckAgain && (
        <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={onCheckAgain} disabled={isChecking}>
          {isChecking ? <Loader2 className="mr-1 h-3 w-3 animate-spin" aria-hidden="true" /> : null}
          Check again
        </Button>
      )}
    </div>
  );
};

const PYANNOTE_LICENCE = 'https://huggingface.co/pyannote/speaker-diarization-community-1';

/** The token's state as a badge (UAT #14): working, rejected, not set. */
const tokenState = (s: HuggingFaceTokenStatus | undefined, error: unknown, isError: boolean): { tone: StateTone; text: string } => {
  if (!s) {
    if (isNeedsKey(error)) return { tone: 'off', text: NEEDS_KEY_MESSAGE };
    return isError ? { tone: 'bad', text: "Couldn't ask the server" } : { tone: 'checking', text: 'Checking…' };
  }
  if (!s.set) return { tone: 'off', text: 'Not set' };
  if (s.check === 'invalid') return { tone: 'bad', text: 'Hugging Face rejects it' };
  if (s.check === 'unknown') return { tone: 'warn', text: "Set, but Hugging Face couldn't be reached to check it" };
  return { tone: 'ok', text: s.account ? `Token working · account ${s.account}` : 'Token working' };
};

/** Settings: whether the server has a token, what needs it, and change or remove it (spec 031, UAT #14). */
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
      setOutcome(outcomeOfError(error));
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
  const state = tokenState(s, status.error, status.isError);
  const working = s?.set && s.check !== 'invalid';
  const canChange = isAdmin !== false && s && s.source !== 'environment';

  return (
    <Card data-testid="huggingface-token">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base">Hugging Face token</CardTitle>
          <StateBadge tone={state.tone} testId="hf-token-state">{state.text}</StateBadge>
        </div>
        <CardDescription>
          Needed by pyannote (who spoke when). Its models are free, but Hugging Face lets an account download them
          only after accepting their licence, so the server needs a token from that account. The token stays on the
          server.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {!working && (
          <ol className="list-decimal space-y-1 pl-5 text-muted-foreground">
            <li>
              Sign in to Hugging Face (or make a free account) and{' '}
              <a href={TOKENS_PAGE} target="_blank" rel="noreferrer" className="underline">
                create a Read token
              </a>
              .
            </li>
            <li>
              With the same account,{' '}
              <a href={PYANNOTE_LICENCE} target="_blank" rel="noreferrer" className="underline">
                accept the pyannote licence
              </a>
              .
            </li>
            <li>Paste the token here. The server checks it with Hugging Face before saving it.</li>
          </ol>
        )}
        {s?.source === 'environment' && (
          <p className="text-xs text-muted-foreground">
            This token comes from the settings the server was started with ({s.envVar}), so it's changed there, not
            here.
          </p>
        )}
        {s?.source === 'saved' && working && <p className="text-xs text-muted-foreground">Saved on the server.</p>}
        {canChange && !editing && !hasConfiguredApiToken() && (
          <p className="text-xs">
            To add or change it from here: <NextStep kind="no-key" />
          </p>
        )}
        {canChange && !editing && hasConfiguredApiToken() && (
          <div className="flex gap-2">
            <Button size="sm" variant={s.set ? 'outline' : 'default'} onClick={() => setEditing(true)}>
              {s.set ? 'Change' : 'Add a token'}
            </Button>
            {s.source === 'saved' && (
              <Button size="sm" variant="outline" onClick={() => remove.mutate()} disabled={remove.isPending}>
                Remove
              </Button>
            )}
          </div>
        )}
        {isAdmin === false && !working && (
          <p className="text-xs text-muted-foreground">Only an administrator of this server can add it: ask whoever runs it.</p>
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
          </form>
        )}
        {outcome && (
          <div className="text-xs">
            <OutcomeLine outcome={outcome} />
          </div>
        )}
      </CardContent>
    </Card>
  );
};
