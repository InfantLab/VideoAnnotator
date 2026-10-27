import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ExternalLink, Eye, EyeOff, KeyRound, Loader2, Stethoscope } from 'lucide-react';
import { apiClient } from '@/api/client';
import { runConnectionDiagnostics, formatDiagnosticReport, type DiagnosticReport } from '@/lib/connectionDiagnostics';
import {
  API_TOKEN_STORAGE_KEY,
  API_URL_STORAGE_KEY,
  checkConnection,
  defaultApiUrl,
  describeApiUrl,
  normalizeApiUrl,
  servedWithApi,
  tokenFormatProblem,
  type ConnectionCheck
} from '@/lib/apiConnection';
import { TokenHelp } from '@/components/TokenHelp';
import { StateBadge, type StateTone } from '@/components/settings/StateBadge';
import type { Connection } from '@/hooks/useConnection';

interface TokenSetupProps {
  /** The saved connection as the rest of the viewer sees it (useConnection). */
  connection?: Connection;
  onTokenConfigured?: () => void;
}

const savedUrl = () => localStorage.getItem(API_URL_STORAGE_KEY) ?? defaultApiUrl();
const savedToken = () => localStorage.getItem(API_TOKEN_STORAGE_KEY) ?? (import.meta.env.VITE_API_TOKEN || '');

/** The saved connection as a badge (UAT #63): whether the key works, whose it is. */
const connectionState = (connection: Connection): { tone: StateTone; text: string } => {
  switch (connection.state) {
    case 'checking':
      return { tone: 'checking', text: 'Checking…' };
    case 'no-server':
      return { tone: 'bad', text: "Can't reach the server" };
    case 'needs-key':
      return { tone: 'warn', text: 'This server needs a key' };
    case 'key-refused':
      return { tone: 'bad', text: 'The server refuses this key' };
    case 'connected':
      if (!connection.user) return { tone: 'ok', text: 'Connected' };
      return {
        tone: 'ok',
        text: `Connected as ${connection.user.name} · ${connection.user.isAdmin ? 'administrator' : 'not an administrator'}`,
      };
  }
};

const PROBLEMS: Record<Exclude<ConnectionCheck, { ok: true }>['problem'], string> = {
  'no-server': "Nothing answered at that address. Check the server is running and the address is right.",
  'needs-key': 'The server answered, but it needs a key. Paste one, or see how to get one below.',
  'key-refused': "The server answered, but it doesn't accept this key. It may have expired or been revoked; get a new one below.",
  'server-error': 'The server answered with an error. Try again, or use Find the problem.',
};

export function TokenSetup({ connection, onTokenConfigured }: TokenSetupProps) {
  const [apiUrl, setApiUrl] = useState(savedUrl);
  const [token, setToken] = useState(savedToken);
  const tokenProblem = tokenFormatProblem(token);
  const [showToken, setShowToken] = useState(false);
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<ConnectionCheck | null>(null);
  const [diagnosticReport, setDiagnosticReport] = useState<DiagnosticReport | null>(null);
  const [isRunningDiagnostics, setIsRunningDiagnostics] = useState(false);

  const changed = apiUrl !== savedUrl() || token !== savedToken();
  const connected = connection?.state === 'connected';

  const save = (url: string, key: string) => {
    const finalUrl = normalizeApiUrl(url);
    localStorage.setItem(API_URL_STORAGE_KEY, finalUrl);
    localStorage.setItem(API_TOKEN_STORAGE_KEY, key);
    apiClient.updateConfig(finalUrl, key);
    onTokenConfigured?.();
    // Every cached query was made with the old address or key.
    setTimeout(() => window.location.reload(), 300);
  };

  // One step (UAT #51): check what was typed, and save it if it works.
  const connect = async () => {
    setChecking(true);
    setResult(null);
    setDiagnosticReport(null);
    try {
      const outcome = await checkConnection(apiUrl, token);
      setResult(outcome);
      if (outcome.ok) save(apiUrl, token.trim());
    } finally {
      setChecking(false);
    }
  };

  const resetToDefaults = () => {
    setApiUrl(defaultApiUrl());
    setToken(import.meta.env.VITE_API_TOKEN || '');
    setResult(null);
  };

  const handleRunDiagnostics = async () => {
    setIsRunningDiagnostics(true);
    setDiagnosticReport(null);
    try {
      const report = await runConnectionDiagnostics(apiUrl, token || undefined);
      setDiagnosticReport(report);
      navigator.clipboard.writeText(formatDiagnosticReport(report)).catch(() => {
        // Clipboard blocked; the report is on the page.
      });
    } catch (error) {
      console.error('Diagnostics failed:', error);
    } finally {
      setIsRunningDiagnostics(false);
    }
  };

  const state = connection ? connectionState(connection) : null;
  const failed = result && result.ok === false ? result : null;
  const help = (
    <>
      <TokenHelp />
      <Button variant="link" size="sm" className="pl-0" asChild>
        <a
          href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/usage/CLIENT_TOKEN_SETUP_GUIDE.md"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1"
        >
          The full guide to keys <ExternalLink className="h-3 w-3" />
        </a>
      </Button>
    </>
  );

  return (
    <Card data-testid="connection-card">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <KeyRound className="h-5 w-5" aria-hidden="true" />
            Server and key
          </CardTitle>
          {state && <StateBadge tone={state.tone} testId="key-state">{state.text}</StateBadge>}
        </div>
        <CardDescription>The VideoAnnotator server this viewer uses, and your key for it.</CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="space-y-2">
          <Label htmlFor="api-url">Server address</Label>
          <div className="flex gap-2">
            <Input
              id="api-url"
              value={apiUrl}
              onChange={(e) => {
                setApiUrl(e.target.value);
                setResult(null);
              }}
              className="flex-1"
            />
            {import.meta.env.DEV && (
              <Button variant="outline" onClick={() => setApiUrl('')} title="Use the dev server's proxy" type="button" disabled={!apiUrl}>
                Use the proxy
              </Button>
            )}
          </div>
          <p className="text-sm text-muted-foreground">
            {apiUrl
              ? 'For example http://127.0.0.1:18011.'
              : import.meta.env.DEV
                ? "Empty: the dev server's proxy to the API."
                : `Empty: the server that served this page (${describeApiUrl('')}).`}
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="api-token">Key</Label>
          <div className="relative">
            <Input
              id="api-token"
              type={showToken ? 'text' : 'password'}
              value={token}
              onChange={(e) => {
                setToken(e.target.value);
                setResult(null);
              }}
              placeholder="va_ followed by 43 characters"
              className="pr-10"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3"
              onClick={() => setShowToken(!showToken)}
              aria-label={showToken ? 'Hide the key' : 'Show the key'}
            >
              {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
          {tokenProblem && (
            <p className="text-sm text-destructive" role="alert">{tokenProblem}</p>
          )}
          {!tokenProblem && !token.trim() && (
            <p className="text-sm text-muted-foreground">Leave it empty only if the server was started without keys.</p>
          )}
        </div>

        {failed && (
          <div role="alert" className="space-y-2 rounded-md border border-amber-500/50 bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            <p>{PROBLEMS[failed.problem]}</p>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={handleRunDiagnostics} disabled={isRunningDiagnostics}>
                <Stethoscope className="mr-1 h-4 w-4" />
                {isRunningDiagnostics ? 'Looking…' : 'Find the problem'}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => save(apiUrl, token.trim())} disabled={Boolean(tokenProblem)}>
                Save anyway
              </Button>
            </div>
          </div>
        )}
        {result?.ok && <p role="status" className="text-sm text-green-700 dark:text-green-400">Connected. Reloading…</p>}

        {diagnosticReport && (
          <Alert>
            <Stethoscope className="h-4 w-4" />
            <AlertTitle>What was tried</AlertTitle>
            <AlertDescription>
              <ul className="mt-2 space-y-1 text-sm">
                {diagnosticReport.results.map((r, i) => (
                  <li key={i}>
                    {r.passed ? 'Worked' : 'Failed'}: {r.test}
                    {r.error && <span className="block text-xs text-destructive">{r.error}</span>}
                  </li>
                ))}
              </ul>
              {diagnosticReport.recommendations.length > 0 && (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-xs">
                  {diagnosticReport.recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              )}
              <p className="mt-2 text-xs text-muted-foreground">Copied, to send to whoever helps you.</p>
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {/* Once connected the badge says so; the button comes back with a change. */}
          {(changed || !connected) && (
            <Button
              onClick={() => void connect()}
              disabled={checking || Boolean(tokenProblem) || (!apiUrl.trim() && !servedWithApi())}
            >
              {checking && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}
              {checking ? 'Checking…' : 'Connect'}
            </Button>
          )}
          {changed && (
            <Button variant="ghost" onClick={() => { setApiUrl(savedUrl()); setToken(savedToken()); setResult(null); }}>
              Undo changes
            </Button>
          )}
          <Button
            variant="ghost"
            onClick={resetToDefaults}
            title={`Address: ${describeApiUrl(defaultApiUrl())}, no key`}
          >
            Reset to defaults
          </Button>
        </div>

        {connected ? (
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground">How to get another key</summary>
            <div className="mt-3 space-y-2">{help}</div>
          </details>
        ) : (
          <div className="space-y-2">
            <h4 id="how-to-get-a-key" className="font-medium">How to get a key</h4>
            {help}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
