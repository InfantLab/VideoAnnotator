import { Loader2, RefreshCw } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useCurrentUser } from "@/hooks/useCurrentUser";
import { MANUAL_RESTART_HINT, useServerRestart } from "@/hooks/useServerRestart";

interface RestartRequiredBannerProps {
  restartRequired: boolean;
}

/**
 * Shown when a pipeline install finished but the server must restart to load it
 * (an expected step, not an error). Offers the restart itself (VideoAnnotator
 * spec 011): the viewer waits for the server to come back, then refreshes, so the
 * user never needs a terminal. Admin-only on the server; non-admins get the reason.
 * See specs/002-pipeline-extras-install (User Story 3, FR-008).
 */
export const RestartRequiredBanner = ({ restartRequired }: RestartRequiredBannerProps) => {
  const { isAdmin } = useCurrentUser();
  const { state, restart, reset } = useServerRestart();

  if (!restartRequired && state.phase !== "restarting") return null;

  if (state.phase === "requesting" || state.phase === "restarting") {
    return (
      <Alert role="status">
        <Loader2 className="h-4 w-4 animate-spin" />
        <AlertTitle>Restarting server&hellip;</AlertTitle>
        <AlertDescription>
          This usually takes a few seconds. The page will update when the server is back.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Alert>
      <RefreshCw className="h-4 w-4" />
      <AlertTitle>Server restart needed</AlertTitle>
      <AlertDescription className="space-y-2">
        <p>
          A pipeline install has finished, but it updated packages the running server had already
          loaded, so the server needs to restart to use it. Newly installed pipelines stay locked
          until then.
        </p>

        {state.phase === "confirm_force" && (
          <div className="space-y-2">
            <p className="font-medium">
              {state.runningJobCount === 1
                ? "1 annotation job is running and would be interrupted."
                : `${state.runningJobCount} annotation jobs are running and would be interrupted.`}{" "}
              Interrupted jobs are marked failed and can be retried.
            </p>
            <div className="flex gap-2">
              <Button size="sm" variant="destructive" onClick={() => restart(true)}>
                Restart anyway
              </Button>
              <Button size="sm" variant="outline" onClick={reset}>
                Wait
              </Button>
            </div>
          </div>
        )}

        {state.phase === "failed" && (
          <div className="space-y-1">
            <p className="font-medium text-destructive">{state.message}</p>
            {state.hint && <p className="text-muted-foreground">{state.hint}</p>}
            <Button size="sm" variant="outline" onClick={() => restart(false)}>
              Try again
            </Button>
          </div>
        )}

        {state.phase === "timed_out" && (
          <p className="font-medium text-destructive">
            The server hasn't come back after two minutes. {MANUAL_RESTART_HINT}
          </p>
        )}

        {(state.phase === "idle" || state.phase === "done") &&
          (isAdmin === false ? (
            <p className="text-muted-foreground">
              Restarting needs an administrator API key. Ask whoever runs your server to restart it.
            </p>
          ) : (
            <Button size="sm" onClick={() => restart(false)}>
              <RefreshCw className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
              Restart server
            </Button>
          ))}
      </AlertDescription>
    </Alert>
  );
};

export default RestartRequiredBanner;
