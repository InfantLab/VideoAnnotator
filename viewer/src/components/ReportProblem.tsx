import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Copy, ExternalLink, Mail } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { getApiBaseUrl } from '@/api/client';
import { useServerVersionShown } from '@/hooks/useServerVersionShown';
import { describeApiUrl } from '@/lib/apiConnection';
import { isDemoVideo } from '@/lib/demos';
import { recentErrors } from '@/lib/errorLog';
import {
  collectDetails,
  detailLines,
  githubIssueUrl,
  mailtoUrl,
  REPORT_EMAIL,
  reportBody,
  type Report,
} from '@/lib/problemReport';

interface ReportProblemDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The video on screen, if any. */
  video?: string | null;
}

const host = () => describeApiUrl(getApiBaseUrl()).replace(/^https?:\/\//, '');

/**
 * Report a problem (UAT #68): the researcher says what happened, sees every
 * detail that goes with it, and sends it from their own mail or GitHub.
 */
export function ReportProblemDialog({ open, onOpenChange, video = null }: ReportProblemDialogProps) {
  const server = useServerVersionShown();
  const [what, setWhat] = useState('');
  const [expected, setExpected] = useState('');
  const [withDetails, setWithDetails] = useState(true);
  // A study's file names can identify people; a demo clip's can't (P1).
  const [withVideo, setWithVideo] = useState(() => isDemoVideo(video));
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (open) setWithVideo(isDemoVideo(video));
  }, [open, video]);

  const details = useMemo(
    () => (open ? collectDetails({ server, address: server ? host() : null, errors: recentErrors() }) : null),
    [open, server],
  );
  const report: Report = {
    what,
    expected,
    details: withDetails ? details : null,
    video: withVideo ? video : null,
  };
  const ready = what.trim().length > 0;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(reportBody(report));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard blocked: email and GitHub still work.
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-xl overflow-y-auto" data-testid="report-problem">
        <DialogHeader>
          <DialogTitle>Report a problem</DialogTitle>
          <DialogDescription>
            Nothing is sent until you send it, from your own email or GitHub.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 text-sm">
          <div className="space-y-1.5">
            <Label htmlFor="report-what">What happened?</Label>
            <Textarea id="report-what" value={what} onChange={(e) => setWhat(e.target.value)} rows={3} autoFocus />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="report-expected">What did you expect? (optional)</Label>
            <Textarea id="report-expected" value={expected} onChange={(e) => setExpected(e.target.value)} rows={2} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Checkbox id="report-details" checked={withDetails} onCheckedChange={(v) => setWithDetails(v === true)} />
              <Label htmlFor="report-details" className="font-normal">
                Add details that help us find it
              </Label>
            </div>
            {withDetails && details && (
              <details className="ml-6 text-xs text-muted-foreground" data-testid="report-details-list">
                <summary className="cursor-pointer">What's added</summary>
                <ul className="mt-2 space-y-0.5">
                  {detailLines(details).map((line) => (
                    <li key={line} className="break-words">
                      {line}
                    </li>
                  ))}
                </ul>
                {video && (
                  <div className="mt-2 flex items-center gap-2">
                    <Checkbox id="report-video" checked={withVideo} onCheckedChange={(v) => setWithVideo(v === true)} />
                    <Label htmlFor="report-video" className="font-normal">
                      The video's name: {video}
                    </Label>
                  </div>
                )}
                <p className="mt-2">Never added: your key, folder paths, results.</p>
              </details>
            )}
          </div>

          <p className="text-xs text-muted-foreground">A screenshot helps: add one to the email or issue once it opens.</p>

          <div className="flex flex-wrap justify-end gap-2">
            <Button variant="outline" size="sm" onClick={() => void copy()} disabled={!ready}>
              <Copy className="mr-1 h-4 w-4" />
              {copied ? 'Copied' : 'Copy'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!ready}
              title={`To ${REPORT_EMAIL}`}
              onClick={() => (window.location.href = mailtoUrl(report))}
            >
              <Mail className="mr-1 h-4 w-4" />
              Email it
            </Button>
            <Button
              size="sm"
              disabled={!ready}
              title="Needs a GitHub account"
              onClick={() => window.open(githubIssueUrl(report), '_blank', 'noopener,noreferrer')}
            >
              Open a GitHub issue
              <ExternalLink className="ml-1 h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** A trigger and its dialog together, so any page can offer it without a provider. */
export function ReportProblemButton({
  children = 'Report a problem',
  className,
  variant = 'link',
  video,
}: {
  children?: ReactNode;
  className?: string;
  variant?: 'link' | 'outline' | 'ghost';
  video?: string | null;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant={variant} size="sm" className={className} onClick={() => setOpen(true)}>
        {children}
      </Button>
      <ReportProblemDialog open={open} onOpenChange={setOpen} video={video} />
    </>
  );
}
