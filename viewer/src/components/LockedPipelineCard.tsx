import { AlertTriangle, CheckCircle2, Info, Lock, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ExtrasInstallJob, PipelineDescriptor, ReadinessItem } from "@/types/pipelines";

interface LockedPipelineCardProps {
  pipeline: PipelineDescriptor;
  /** Badge text for why it can't be selected (default "Not installed"). */
  badge?: string;
  /**
   * The extras group that installs this pipeline, for "Installs <group> (approx.
   * N MB), which also enables ..." (VideoAnnotator spec 011). Without it (older
   * servers) the raw install hint is shown instead.
   */
  group?: { name: string; approxMb: number | null; alsoEnables: string[] };
  /** Install action / progress / failure content for this pipeline's extras group. Omitted when there's nothing actionable to show (e.g. no session token configured, or the hint couldn't be parsed into an extras group name). */
  children?: React.ReactNode;
}

/**
 * Renders a pipeline the server knows about but that can't be selected yet
 * (not installed, installing, needs a restart or setup), per
 * specs/002-pipeline-extras-install (User Story 1) and VideoAnnotator spec 011:
 * visible and explained, not silently omitted from the pipeline list.
 */
export const LockedPipelineCard = ({ pipeline, badge = "Not installed", group, children }: LockedPipelineCardProps) => {
  const description = pipeline.description ?? "No description provided.";

  return (
    // Not aria-disabled/cursor-not-allowed: the card isn't a control, and both
    // leak onto the Install button inside it (assistive tech then announces a
    // working button as unavailable). The badge carries it.
    <div data-locked="true" className="flex flex-col gap-1 rounded-lg border border-dashed border-border bg-muted/40 p-3 opacity-90">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <Lock className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <span className="text-sm font-medium text-foreground">{pipeline.name}</span>
        </div>
        <Badge variant="outline" className="text-[10px] px-1.5 py-0">
          {badge}
        </Badge>
      </div>
      <p className="text-xs text-muted-foreground">{description}</p>
      {group ? (
        <p className="text-xs text-muted-foreground">
          Installs the <span className="font-medium text-foreground">{group.name}</span> group
          {group.approxMb != null && <> (approx. {formatMb(group.approxMb)})</>}
          {group.alsoEnables.length > 0 && <>, which also enables {group.alsoEnables.join(", ")}</>}.
        </p>
      ) : (
        pipeline.installHint && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">To enable:</span>{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono">{pipeline.installHint}</code>
          </p>
        )
      )}
      {children}
    </div>
  );
};

const formatMb = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

interface ReadinessDetailsProps {
  /** Why the pipeline can't be used yet. */
  blockers?: ReadinessItem[];
  /** Worth knowing, never blocking (licences, first-run downloads). */
  notes?: ReadinessItem[];
  /** Re-check the server, e.g. after starting Ollama or setting a token. */
  onCheckAgain?: () => void;
  isChecking?: boolean;
}

/**
 * Blockers and notes from a pipeline's readiness (VideoAnnotator spec 011).
 * Display only: secrets are the server's container configuration, so the viewer
 * says what to set and where, but never takes a value.
 */
export const ReadinessDetails = ({ blockers = [], notes = [], onCheckAgain, isChecking }: ReadinessDetailsProps) => {
  if (blockers.length === 0 && notes.length === 0) return null;
  return (
    <div className="mt-1 space-y-1">
      {blockers.map((b) => (
        <p key={`${b.kind}:${b.name}`} className="flex items-start gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-500">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>
            {b.message}
            {b.kind === "import_error" && <> The server administrator needs to look at this.</>}
            {b.helpUrl && (
              <>
                {" "}
                <a href={b.helpUrl} target="_blank" rel="noreferrer" className="underline">
                  {b.kind === "secret" ? "Get a token" : b.name === "ollama" ? "Get Ollama" : "More"}
                </a>
              </>
            )}
          </span>
        </p>
      ))}
      {notes.map((n) => (
        <p key={`${n.kind}:${n.name}`} className="flex items-start gap-1.5 text-xs text-muted-foreground">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>
            {n.message}
            {n.helpUrl && (
              <>
                {" "}
                <a href={n.helpUrl} target="_blank" rel="noreferrer" className="underline">
                  {n.kind === "licence" ? "Open the model page" : "More"}
                </a>
              </>
            )}
          </span>
        </p>
      ))}
      {blockers.length > 0 && onCheckAgain && (
        <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" onClick={onCheckAgain} disabled={isChecking}>
          {isChecking ? <Loader2 className="mr-1 h-3 w-3 animate-spin" aria-hidden="true" /> : null}
          Check again
        </Button>
      )}
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

  if (job?.status === 'completed' && job.activation === 'live') {
    // Usable already (VideoAnnotator spec 011); the catalog refresh that
    // useExtrasInstall triggers swaps this card for a selectable one.
    return (
      <p className="mt-1 flex items-center gap-1.5 text-xs font-medium text-green-700 dark:text-green-500">
        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
        Installed. Loading&hellip;
      </p>
    );
  }

  if (job?.status === 'completed') {
    // Needs a restart (or the server predates spec 011's live activation).
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
