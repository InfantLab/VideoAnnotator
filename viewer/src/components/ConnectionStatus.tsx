import { CheckCircle2, KeyRound, Loader2, Settings, XCircle } from 'lucide-react';
import { Link } from 'react-router-dom';

import { getApiBaseUrl } from '@/api/client';
import { NEEDS_KEY_MESSAGE } from '@/api/handleError';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useConnection, type ConnectionState } from '@/hooks/useConnection';
import { describeApiUrl } from '@/lib/apiConnection';

const LABEL: Record<ConnectionState, string> = {
  checking: 'Checking…',
  'no-server': 'No server',
  'needs-key': 'Connect with your key',
  'key-refused': 'Key not accepted',
  connected: 'Connected',
};

function StateIcon({ state }: { state: ConnectionState }) {
  if (state === 'checking') return <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />;
  if (state === 'connected') return <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />;
  if (state === 'needs-key') return <KeyRound className="h-3.5 w-3.5 text-amber-500" />;
  return <XCircle className="h-3.5 w-3.5 text-red-600" />;
}

/** The nav's connection badge (UAT #17, #38): a checked state, with the details one click away. */
export function ConnectionStatus() {
  const { state, user, version } = useConnection();
  const server = describeApiUrl(getApiBaseUrl());

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 gap-1.5 px-2" title={LABEL[state]} aria-label={LABEL[state]} data-testid="connection-status">
          <StateIcon state={state} />
          <span className="hidden lg:inline">{LABEL[state]}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-3 text-sm">
        <div className="flex items-center gap-2 font-medium">
          <StateIcon state={state} />
          {LABEL[state]}
        </div>
        <div className="text-muted-foreground">
          <p className="break-all">Server: {server}</p>
          {version && <p>VideoAnnotator v{version}</p>}
        </div>
        {state === 'connected' && (
          <p>
            {user
              ? `Signed in as ${user.name}${user.isAdmin ? ', an administrator' : ''}.`
              : 'This server needs no key.'}
          </p>
        )}
        {state === 'needs-key' && <p>{NEEDS_KEY_MESSAGE}</p>}
        {state === 'key-refused' && (
          <p>
            The server didn't accept your key: it may have expired or been withdrawn. Ask whoever runs the server for a
            new one.
          </p>
        )}
        {state === 'no-server' && <p>Can't reach the server. Check it's running, and the address in Settings.</p>}
        <Button asChild size="sm" variant={state === 'connected' ? 'outline' : 'default'}>
          <Link to="/settings">
            <Settings className="mr-1 h-3.5 w-3.5" />
            {state === 'needs-key' || state === 'key-refused' ? 'Connect with your key' : 'Settings'}
          </Link>
        </Button>
      </PopoverContent>
    </Popover>
  );
}
