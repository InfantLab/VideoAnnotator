import { ReactNode, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { AlertCircle, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { parseApiError } from "@/lib/errorHandling";
import { ErrorDisplay } from "@/components/ErrorDisplay";
import { useRefreshPipelineCatalog, useExtrasGroups } from "@/hooks/usePipelineCatalog";
import { LockedPipelineCard, ExtrasInstallStatus, ReadinessDetails, RestoringStatus } from "@/components/LockedPipelineCard";
import { RestartRequiredBanner } from "@/components/RestartRequiredBanner";
import { RestartToActivate } from "@/components/RestartToActivate";
import type { PipelineDescriptor } from "@/types/pipelines";
import { useExtrasInstall } from "@/hooks/useExtrasInstall";
import { useCurrentUser } from "@/hooks/useCurrentUser";
import { extrasGroupOf, isPipelineSelectable, notReadyReason, PIPELINE_CARD_BADGE, pipelineCardMode } from "@/lib/pipelineExtras";
import { hasConfiguredApiToken } from "@/api/client";
import { APIError } from "@/api/handleError";
import { PipelineAbout, PipelineTitle } from '@/components/PipelineGuide';
import { pipelineSummary } from '@/lib/pipelineGuide';

/**
 * Which pipelines to run, with the guide to each (spec 026), and installing
 * the ones not installed yet: New run's "What to find" and the Pipelines page.
 */
/** The server's pipeline families, in the researcher's words (UAT #19, P7). */
const GROUP_NAMES: Record<string, string> = {
  audio: 'Sound',
  face: 'Faces',
  person: 'People',
  scene: 'Scenes',
  vlm: 'Ask AI',
};
const groupRank = (group: string) => {
  const at = Object.keys(GROUP_NAMES).indexOf(group);
  return at < 0 ? Object.keys(GROUP_NAMES).length : at;
};

export const PipelineSelectionStep = ({
  pipelines,
  selectedPipelines,
  setSelectedPipelines,
  isLoading,
  error,
  onRetry,
  restartRequired = false,
  selectionNotices = [],
  onDismissNotices,
  presetBar,
  manage = false,
  settingsCue,
  renderSettings
}: {
  pipelines: PipelineDescriptor[];
  selectedPipelines: string[];
  setSelectedPipelines: (pipelines: string[]) => void;
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  restartRequired?: boolean;
  selectionNotices?: string[];
  onDismissNotices?: () => void;
  /** The workflow picker; a slot so the step stays usable without a server that has them. */
  presetBar?: ReactNode;
  /** Install and check pipelines without choosing any for a run (the Pipelines page). */
  manage?: boolean;
  /** A chosen pipeline's settings in a few words, on its row ("defaults", "Age on"). */
  settingsCue?: (pipelineId: string) => string;
  /** A chosen pipeline's settings, opened under its row (UAT #35). */
  renderSettings?: (pipelineId: string) => ReactNode;
}) => {
  const [open, setOpen] = useState<Set<string>>(new Set());
  const toggleOpen = (id: string) =>
    setOpen((now) => {
      const next = new Set(now);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const groupedPipelines = useMemo(() => {
    const groups = new Map<string, PipelineDescriptor[]>();
    pipelines.forEach((pipeline) => {
      const groupKey = pipeline.group || 'Other';
      const current = groups.get(groupKey) ?? [];
      current.push(pipeline);
      groups.set(groupKey, current);
    });

    const sorted = Array.from(groups.entries())
      .map(([groupName, list]) => ({
        groupName,
        list: list.slice().sort((a, b) => a.name.localeCompare(b.name))
      }))
      .sort((a, b) => groupRank(a.groupName) - groupRank(b.groupName) || a.groupName.localeCompare(b.groupName));
    // New run: one list, the kind's icon on each row instead of headings (UAT #35).
    return manage ? sorted : [{ groupName: '', list: sorted.flatMap((g) => g.list) }];
  }, [pipelines, manage]);

  const blockedSelection = pipelines.filter(
    (p) => selectedPipelines.includes(p.id) && !isPipelineSelectable(p)
  );

  const togglePipeline = (pipelineId: string) => {
    if (selectedPipelines.includes(pipelineId)) {
      setSelectedPipelines(selectedPipelines.filter((p) => p !== pipelineId));
    } else {
      setSelectedPipelines([...selectedPipelines, pipelineId]);
    }
  };

  const canInstallExtras = hasConfiguredApiToken();
  const { isAdmin } = useCurrentUser();
  const { jobsByExtra, install, isInstalling, installError, installErrorExtraName, adoptJob } =
    useExtrasInstall(pipelines);
  const { data: extrasGroups } = useExtrasGroups({ enabled: pipelines.length > 0 });
  const refreshCatalog = useRefreshPipelineCatalog();
  const [isCheckingAgain, setIsCheckingAgain] = useState(false);

  // Pipelines sharing one extras group (e.g. two locked pipelines both unlocked by
  // `face`) must be triggered/tracked together, not treated as independent installs.
  const lockedPipelineIdsByExtra = useMemo(() => {
    const map = new Map<string, string[]>();
    pipelines.forEach((pipeline) => {
      if (pipelineCardMode(pipeline) !== 'selectable') {
        const extraName = extrasGroupOf(pipeline);
        if (extraName) {
          map.set(extraName, [...(map.get(extraName) ?? []), pipeline.id]);
        }
      }
    });
    return map;
  }, [pipelines]);

  // An install the server reports as running but this browser didn't start
  // (another tab, another admin): track it so the card shows its progress.
  useEffect(() => {
    pipelines.forEach((pipeline) => {
      const readiness = pipeline.readiness;
      const extraName = extrasGroupOf(pipeline);
      if ((readiness?.state === 'installing' || readiness?.state === 'restoring') && readiness.installJobId && extraName) {
        adoptJob(extraName, readiness.installJobId, lockedPipelineIdsByExtra.get(extraName) ?? [pipeline.id]);
      }
    });
  }, [pipelines, adoptJob, lockedPipelineIdsByExtra]);

  // Setup is done outside the viewer (start Ollama, set a token in the server's
  // env), so offer a re-check. The server refreshes a stale service check in the
  // background, so fetch twice: the second read picks up the fresh result.
  const checkAgain = async () => {
    setIsCheckingAgain(true);
    try {
      await refreshCatalog({ forceServerRefresh: true });
      await new Promise((resolve) => setTimeout(resolve, 1500));
      await refreshCatalog({ forceServerRefresh: true });
    } catch {
      // The catalog query surfaces its own errors.
    } finally {
      setIsCheckingAgain(false);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-full" />
      </div>
    );
  }

  if (error) {
    const parsed = parseApiError(error);
    // If we have a network error but no pipelines, it might be because we haven't connected yet
    // Don't show "undefined" or confusing errors
    if (parsed.message === 'undefined' || !parsed.message) {
        parsed.message = 'Failed to load pipelines. Please check your connection.';
    }
    
    return (
      <ErrorDisplay error={parsed} />
    );
  }

  if (pipelines.length === 0) {
    return (
      <Alert>
        <AlertTitle>No pipelines detected</AlertTitle>
        <AlertDescription>
          Connect to a VideoAnnotator v1.2.x server and refresh the catalog before creating jobs.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <RestartRequiredBanner restartRequired={restartRequired} />

      {presetBar}

      {selectionNotices.length > 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <div className="space-y-1">
              {selectionNotices.map((notice) => (
                <p key={notice}>{notice}</p>
              ))}
            </div>
            {onDismissNotices && (
              <Button type="button" size="sm" variant="ghost" className="mt-1 h-6 px-2 text-xs" onClick={onDismissNotices}>
                Dismiss
              </Button>
            )}
          </AlertDescription>
        </Alert>
      )}

      {pipelines.some((pipeline) => pipelineCardMode(pipeline) === 'install') && (
        <p className="text-xs text-muted-foreground">
          {isAdmin === true && (
            <>Pipelines marked "Not installed" below can be installed from here — your API key has admin access.</>
          )}
          {isAdmin === false && (
            <>
              Pipelines marked "Not installed" below require an <strong>admin-scoped</strong> API
              key to install, which yours doesn't have. See "Your key" on{" "}
              <Link to="/help" className="underline">
                Help
              </Link>{" "}
              for how to get one.
            </>
          )}
          {isAdmin === 'unknown' && (
            <>
              Pipelines marked "Not installed" below can be installed from here, but only with an{" "}
              <strong>admin-scoped</strong> API key. If you're running your own single-user
              server, your key is usually an administrator&apos;s already: whether it is shows on{" "}
              <Link to="/settings" className="underline">
                Settings
              </Link>
              .
            </>
          )}
        </p>
      )}

      <div className="space-y-6">
        {groupedPipelines.map(({ groupName, list }) => (
          <div key={groupName} className="space-y-2">
            {groupName && (
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-foreground">{GROUP_NAMES[groupName] ?? groupName}</h3>
              </div>
            )}
            <div className={manage ? "space-y-3" : "space-y-2"}>
              {list.map((pipeline) => {
                const mode = pipelineCardMode(pipeline);
                if (mode !== 'selectable') {
                  const extraName = extrasGroupOf(pipeline);
                  const job = extraName ? jobsByExtra[extraName] : undefined;
                  const triggering = extraName ? isInstalling(extraName) : false;
                  const rawError =
                    extraName && installErrorExtraName === extraName ? installError : null;
                  const triggerError = rawError
                    ? {
                        status: rawError instanceof APIError ? rawError.status : undefined,
                        message: rawError.message
                      }
                    : null;

                  const group =
                    pipeline.readiness && extraName && (mode === 'install' || mode === 'installing')
                      ? {
                          name: extraName,
                          approxMb: extrasGroups?.get(extraName)?.approxDownloadMb ?? null,
                          alsoEnables: pipelines
                            .filter((p) => p.id !== pipeline.id && extrasGroupOf(p) === extraName)
                            .map((p) => p.name)
                        }
                      : undefined;

                  return (
                    <LockedPipelineCard
                      key={pipeline.id}
                      pipeline={pipeline}
                      badge={PIPELINE_CARD_BADGE[mode]}
                      group={group}
                    >
                      {selectedPipelines.includes(pipeline.id) && (
                        // Not ready means it can't be chosen, but one that's
                        // already selected must always be removable.
                        <div className="mt-1 flex flex-wrap items-center gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-xs text-amber-800 dark:text-amber-400">
                          <span>Selected, but it can&apos;t run: {notReadyReason(pipeline)}.</span>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-6 px-2 text-[11px]"
                            onClick={() => togglePipeline(pipeline.id)}
                          >
                            Remove from this run
                          </Button>
                        </div>
                      )}
                      {mode === 'restoring' && <RestoringStatus />}
                      {mode === 'restart' && (!job || job.status !== 'completed') && (
                        <RestartToActivate />
                      )}
                      {(mode === 'setup' || mode === 'unavailable') && (
                        <ReadinessDetails
                          blockers={pipeline.readiness?.blockers}
                          notes={pipeline.readiness?.notes}
                          onCheckAgain={checkAgain}
                          isChecking={isCheckingAgain}
                          isAdmin={isAdmin}
                        />
                      )}
                      {canInstallExtras && extraName && (mode === 'install' || mode === 'installing' || (mode === 'restart' && job?.status === 'completed')) && (
                        <ExtrasInstallStatus
                          job={job}
                          isTriggering={triggering || (mode === 'installing' && !job)}
                          triggerError={triggerError}
                          adminStatus={isAdmin}
                          onInstall={() => {
                            // Errors are surfaced via `installError`/`installErrorExtraName`
                            // state (rendered above) - swallow the rejection here so it
                            // doesn't also surface as an unhandled promise rejection.
                            install(extraName, lockedPipelineIdsByExtra.get(extraName) ?? [pipeline.id]).catch(
                              () => {}
                            );
                          }}
                        />
                      )}
                    </LockedPipelineCard>
                  );
                }

                const checked = selectedPipelines.includes(pipeline.id);
                const description = pipelineSummary(pipeline);
                if (manage) {
                  return (
                    <div key={pipeline.id} className="flex flex-col gap-1 rounded-lg border border-border bg-card p-3">
                      <div className="flex items-center justify-between gap-2">
                        <PipelineTitle pipeline={pipeline} />
                        <div className="flex flex-wrap items-center gap-1.5">
                          <Badge className="text-[10px] px-1.5 py-0 bg-green-600 hover:bg-green-600">Ready</Badge>
                          {pipeline.version && (
                            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                              v{pipeline.version}
                            </Badge>
                          )}
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground">{description}</p>
                      <ReadinessDetails notes={pipeline.readiness?.notes} />
                      <PipelineAbout pipeline={pipeline} defaultOpen />
                    </div>
                  );
                }
                const cue = checked ? settingsCue?.(pipeline.id) : undefined;
                const opened = checked && open.has(pipeline.id);
                return (
                  <div
                    key={pipeline.id}
                    data-testid="pipeline-row"
                    className={`rounded-lg border transition ${checked ? 'border-primary bg-primary/5' : 'border-border bg-card hover:border-primary/70'}`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-2">
                      <label className="flex min-w-0 cursor-pointer items-center gap-3" title={description}>
                        <input
                          type="checkbox"
                          className="rounded"
                          checked={checked}
                          onChange={() => togglePipeline(pipeline.id)}
                        />
                        <PipelineTitle pipeline={pipeline} />
                      </label>
                      {cue && renderSettings && (
                        <button
                          type="button"
                          className="ml-auto flex min-w-0 max-w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                          aria-expanded={opened}
                          aria-label={`${pipeline.name} settings: ${cue}`}
                          onClick={() => toggleOpen(pipeline.id)}
                          data-testid="settings-cue"
                        >
                          <span className={`truncate ${cue === 'defaults' ? '' : 'text-foreground'}`}>{cue}</span>
                          <ChevronRight className={`h-3.5 w-3.5 shrink-0 transition-transform ${opened ? 'rotate-90' : ''}`} aria-hidden="true" />
                        </button>
                      )}
                    </div>
                    <div className="px-3">
                      <ReadinessDetails notes={pipeline.readiness?.notes} />
                    </div>
                    {opened && renderSettings && (
                      <div className="border-t px-4 pb-2" data-testid="pipeline-settings">
                        {renderSettings(pipeline.id)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {blockedSelection.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Remove {blockedSelection.map((p) => p.name).join(", ")} to continue:{" "}
            {blockedSelection.length === 1 ? "it can't" : "they can't"} run right now.
          </AlertDescription>
        </Alert>
      )}

      {!manage && selectedPipelines.length === 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Please select at least one pipeline to proceed.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};
