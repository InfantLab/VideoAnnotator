// A methods paragraph for a paper (spec 025): built by the server from the
// provenance recorded in the result files, shown here to copy or download.
// Pipeline-supplied text (prompts) is rendered as text, never as HTML.

import { useEffect, useState } from 'react';
import { Check, Copy, Download, Loader2 } from 'lucide-react';
import { apiClient, type ExportScope, type MethodsDescription } from '@/api/client';
import { userMessage } from '@/lib/userMessage';
import { methodsMarkdown } from '@/lib/exports';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface MethodsDialogProps {
  scope: Exclude<ExportScope, { kind: 'dataset' }>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setFailed(true);
    }
  };
  return (
    <Button variant="outline" size="sm" onClick={copy} aria-label={`Copy ${label}`}>
      {copied ? <Check className="h-4 w-4 mr-2" /> : <Copy className="h-4 w-4 mr-2" />}
      {copied ? 'Copied' : failed ? 'Select and copy it' : `Copy ${label}`}
    </Button>
  );
}

export default function MethodsDialog({ scope, open, onOpenChange }: MethodsDialogProps) {
  const [methods, setMethods] = useState<MethodsDescription | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  // The scope's value, not its identity: callers build it inline.
  const scopeKey = JSON.stringify(scope);
  useEffect(() => {
    let cancelled = false;
    setMethods(null);
    setProblem(null);
    apiClient
      .getMethods(JSON.parse(scopeKey) as ExportScope)
      .then((m) => !cancelled && setMethods(m))
      .catch((e) => !cancelled && setProblem(userMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [scopeKey]);

  const download = () => {
    if (!methods) return;
    const url = URL.createObjectURL(new Blob([methodsMarkdown(methods)], { type: 'text/markdown' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'methods.md';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto [&>*]:min-w-0">
        <DialogHeader>
          <DialogTitle>Methods paragraph</DialogTitle>
          <DialogDescription>
            Built from what each result file records about how it was made. Citations are{' '}
            <code>[@key]</code> keys for the BibTeX below. Anything not recorded says so.
          </DialogDescription>
        </DialogHeader>
        {!methods && !problem && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
            <Loader2 className="h-4 w-4 animate-spin" /> Writing the methods…
          </p>
        )}
        {problem && (
          <p className="text-sm text-destructive" role="alert">
            Couldn&apos;t write the methods: {problem}
          </p>
        )}
        {methods && (
          <div className="space-y-4">
            <section className="space-y-2">
              <p className="text-sm leading-relaxed whitespace-pre-wrap" data-testid="methods-prose">
                {methods.prose}
              </p>
              <div className="flex flex-wrap gap-2">
                <CopyButton text={methods.prose} label="paragraph" />
                <Button variant="outline" size="sm" onClick={download}>
                  <Download className="h-4 w-4 mr-2" /> Download (.md)
                </Button>
              </div>
            </section>
            {methods.revisions.length > 0 && (
              <section className="space-y-1">
                <h3 className="text-sm font-medium">Model revisions</h3>
                <ul className="list-disc pl-5 text-xs font-mono break-all">
                  {methods.revisions.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </section>
            )}
            <section className="space-y-2">
              <h3 className="text-sm font-medium">References (BibTeX)</h3>
              <pre className="max-h-64 overflow-auto rounded bg-muted p-2 text-xs" data-testid="methods-bibtex">
                {methods.bibtex}
              </pre>
              <CopyButton text={methods.bibtex} label="BibTeX" />
            </section>
            {methods.prompts.map((prompt) => (
              <section key={prompt.sha256} className="space-y-1">
                <h3 className="text-sm font-medium">Prompt {prompt.sha256.slice(0, 12)}</h3>
                <pre className="max-h-48 overflow-auto rounded bg-muted p-2 text-xs whitespace-pre-wrap">
                  {prompt.text}
                </pre>
              </section>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
