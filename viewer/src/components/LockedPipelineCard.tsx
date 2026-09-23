import { AlertTriangle, CheckCircle2, Lock, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ExtrasInstallJob, PipelineDescriptor } from "@/types/pipelines";

interface LockedPipelineCardProps {
  pipeline: PipelineDescriptor;
  /** Install action / progress / failure content for this pipeline's extras group. Omitted when there's nothing actionable to show (e.g. no session token configured, or the hint couldn't be parsed into an extras group name). */
  children?: React.ReactNode;
}

/**
 * Renders a pipeline the server knows about but hasn't installed, per
 * specs/002-pipeline-extras-install (User Story 1): visible and explained,
 * not silently omitted from the pipeline list.
 */
export const LockedPipelineCard = ({ pipeline, children }: LockedPipelineCardProps) => {
  const description = pipeline.description ?? "No description provided.";

  return (
    // Not aria-disabled/cursor-not-allowed: the card isn't a control, and both
    // leak onto the Install button inside it (assistive tech then announces a
    // working button as unavailable). The "Not installed" badge carries it.
    <div data-locked="true" className="flex flex-col gap-1 rounded-lg border border-dashed border-border bg-muted/40 p-3 opacity-90">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <Lock className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <span className="text-sm font-medium text-foreground">{pipeline.name}</span>
        </div>
        <Badge variant="outline" className="text-[10px] px-1.5 py-0">
          Not installed
        </Badge>
      </div>
      <p className="text-xs text-muted-foreground">{description}</p>
      {pipeline.installHint && (
        <p className="text-xs text-muted-foreground">
          <span className="font-medium text-foreground">To enable:</span>{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono">{pipeline.installHint}</code>
        </p>
      )}
      {children}
    </div>
  );
};

interface ExtrasInstallStatusProps {
  job: ExtrasInstallJob | undefined;
  isTriggering: boolean;
  /** Set when the most recent trigger attempt for this extras group failed (e.g. 401/403/422/network). */
  triggerError?: { status?: number; message: string } | null;
  onInstall: () => void;
  /**
   * The current session's admin status, from GET /api/v1/auth/me
   * (useCurrentUser). `false` disables the Install action up front with an
   * explanation, instead of letting the user click through to a bare 403.
   * `'unknown'` (no token, endpoint unsupported by an older server, or still
   * loading) falls back to offering the action and handling its own 403.
   */
  adminStatus: boolean | 'unknown';
}

/**
 * Install button + in-flight/failed status for one locked pipeline's extras
 * group, rendered as a LockedPipelineCard child. Deliberately does not render
 * a "ready to use" state on completion - that's gated on a server restart
 * (see RestartRequiredBanner), not on this job reaching `completed`.
 */
export const ExtrasInstallStatus = ({
  job,
  isTriggering,
  triggerError,
  onInstall,
  adminStatus
}: ExtrasInstallStatusProps) => {
  if (adminStatus === false) {
    return (
      <div className="mt-1 space-y-1">
        <Button type="button" size="sm" variant="secondary" className="h-7 w-fit px-2 text-xs" disabled>
          Install
        </Button>
        <p className="text-xs text-muted-foreground">
          Requires an administrator API key, which yours doesn't have. See the "Getting Help" tab
          on the{" "}
          <Link to="/settings" className="underline">
            Settings
          </Link>{" "}
          page, or ask whoever runs your server to grant one with{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono">
            uv run videoannotator generate-token --admin
          </code>
          .
        </p>
      </div>
    );
  }

  if (triggerError?.status === 403) {
    return (
      <div className="mt-1 space-y-1">
        <p className="text-xs font-medium text-amber-700 dark:text-amber-500">
          Administrator privileges are required to install pipeline extras.
        </p>
        <p className="text-xs text-muted-foreground">
          Running your own single-user server? The token it generated on first startup is usually
          admin already. See the "Getting Help" tab on the{" "}
          <Link to="/settings" className="underline">
            Settings
          </Link>{" "}
          page for how to check or get an admin-scoped token.
        </p>
      </div>
    );
  }

  if (triggerError) {
    return (
      <div className="mt-1 flex items-center gap-2">
        <p className="text-xs font-medium text-destructive">
          Couldn't start the install: {triggerError.message}
        </p>
        <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={onInstall}>
          Retry
        </Button>
      </div>
    );
  }

  if (isTriggering || job?.status === 'pending' || job?.status === 'running') {
    return (
      <p className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
        Installing&hellip; this can take several minutes (larger pipelines download GPU-accelerated
        dependencies).
      </p>
    );
  }

  if (job?.status === 'failed') {
    return (
      <div className="mt-1 space-y-1">
        <p className="flex items-center gap-1.5 text-xs font-medium text-destructive">
          <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          Install failed.
        </p>
        {job.commandOutput && (
          <details className="rounded border border-destructive/30 bg-destructive/5 p-2">
            <summary className="cursor-pointer text-[11px] text-destructive">Show install output</summary>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[11px] text-muted-foreground">
              {job.commandOutput}
            </pre>
          </details>
        )}
        <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={onInstall}>
          Retry install
        </Button>
      </div>
    );
  }

  if (job?.status === 'completed') {
    // The pipeline stays locked until the server restarts and a later catalog
    // fetch reports it available. Say so on the card too: the step-level
    // RestartRequiredBanner can be scrolled out of view, and a card that just
    // reverts to "Not installed" reads as if the install did nothing.
    return (
      <p className="mt-1 flex items-center gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-500">
        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
        Installed. Restart the server to activate it.
      </p>
    );
  }

  return (
    <Button type="button" size="sm" variant="secondary" className="mt-1 h-7 w-fit px-2 text-xs" onClick={onInstall}>
      Install
    </Button>
  );
};

export default LockedPipelineCard;
