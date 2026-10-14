import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { GENERATE_TOKEN_COMMAND } from '@/lib/apiConnection';

function CopyableCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(command).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      },
      () => {
        // Clipboard blocked (e.g. not a secure context); the command is still visible to select.
      }
    );
  };

  return (
    <span className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 font-mono text-xs">
      {command}
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="h-5 w-5 p-0"
        onClick={copy}
        aria-label={`Copy "${command}"`}
        title="Copy"
      >
        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      </Button>
    </span>
  );
}

/** How to get an API key, in the order a new user meets the options. */
export function TokenHelp() {
  return (
    <ol className="list-decimal space-y-3 pl-5 text-sm text-muted-foreground">
      <li>
        <span className="font-medium text-foreground">First start of the server.</span> Its console prints{' '}
        <code className="rounded bg-muted px-1">[API KEY] VIDEOANNOTATOR API KEY GENERATED</code>, then a key
        (<code className="rounded bg-muted px-1">va_</code> followed by 43 characters) and a one-click link,{' '}
        <code className="rounded bg-muted px-1">http://127.0.0.1:18011/viewer-connect?token=…</code>. Open the link:
        it connects this viewer, with nothing to paste.
      </li>
      <li>
        <span className="font-medium text-foreground">Missed it, or need another key.</span> In a terminal on the
        server machine, run <CopyableCommand command={GENERATE_TOKEN_COMMAND} /> (in a source checkout,{' '}
        <code className="rounded bg-muted px-1">uv run {GENERATE_TOKEN_COMMAND}</code>). It prints a new key and a
        one-click link. A key is shown only once, when it's made.
      </li>
      <li>
        <span className="font-medium text-foreground">Someone else runs the server.</span> Ask them to run{' '}
        <code className="rounded bg-muted px-1">{GENERATE_TOKEN_COMMAND} --user &lt;your email&gt;</code> and send you
        the key or the link.
      </li>
      <li>
        <span className="font-medium text-foreground">Server started with authentication off</span>{' '}
        (<code className="rounded bg-muted px-1">AUTH_REQUIRED=false</code>). Leave the key empty.
      </li>
    </ol>
  );
}
