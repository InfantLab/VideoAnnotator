import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ExternalLink, Play } from 'lucide-react';

import { APIError } from '@/api/handleError';
import { NeedsServer } from '@/components/NeedsServer';
import { PageHelp } from '@/components/PageHelp';
import { ReportProblemButton } from '@/components/ReportProblem';
import { AskAIPanel } from '@/components/pipelines/AskAIPanel';
import { PipelineEntry } from '@/components/pipelines/PipelineEntry';
import { PipelineSelectionStep } from '@/components/run/PipelineChoice';
import { WorkflowsSection } from '@/components/workflows/WorkflowsSection';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { usePipelineCatalog, useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';
import { useWorkflows } from '@/hooks/useWorkflows';
import { useSaveYourDefaults, useYourDefaults } from '@/hooks/useYourDefaults';
import { GITHUB_URL } from '@/utils/version';

type Updater = (prev: Record<string, unknown>) => Record<string, unknown>;

const ASK_AI = 'vlm_annotation';

/** Your defaults, edited here and saved a moment after the last change. */
function useEditedDefaults() {
  const query = useYourDefaults();
  const save = useSaveYourDefaults();
  const [yours, setYours] = useState<Record<string, unknown> | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const { mutate } = save;

  useEffect(() => {
    if (yours === null && query.data) setYours(query.data.config);
  }, [yours, query.data]);
  useEffect(() => () => clearTimeout(timer.current), []);

  const onYours = useCallback(
    (updater: Updater) =>
      setYours((prev) => {
        const next = updater(prev ?? {});
        clearTimeout(timer.current);
        timer.current = setTimeout(() => mutate(next as Record<string, Record<string, unknown>>), 600);
        return next;
      }),
    [mutate],
  );
  return { yours: yours ?? {}, onYours, save };
}

/**
 * What each pipeline finds and how it's set, day to day, with installing as
 * one part of that; Ask AI set apart; workflows in their own tab (UAT #71).
 */
export default function Pipelines() {
  const location = useLocation();
  const navigate = useNavigate();
  const { data, isLoading, error } = usePipelineCatalog();
  const refresh = useRefreshPipelineCatalog();
  const workflows = useWorkflows();
  const { yours, onYours, save } = useEditedDefaults();
  const tab = location.hash === '#workflows' ? 'workflows' : 'pipelines';

  const all = data?.catalog.pipelines ?? [];
  const askAI = all.find((p) => p.id === ASK_AI);
  const others = all.filter((p) => p.id !== ASK_AI);
  const step = (pipelines: typeof all, headings = true) => (
    <PipelineSelectionStep
      manage
      headings={headings}
      pipelines={pipelines}
      selectedPipelines={[]}
      setSelectedPipelines={() => {}}
      isLoading={isLoading}
      error={error}
      onRetry={() => refresh({ forceServerRefresh: true })}
      restartRequired={headings && (data?.restartRequired ?? false)}
      renderReady={(pipeline) => <PipelineEntry pipeline={pipeline} yours={yours} onYours={onYours} />}
    />
  );

  return (
    <div className="container mx-auto max-w-4xl space-y-6 px-6 py-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-3xl font-bold">Pipelines</h1>
        <Button asChild>
          <Link to="/run">
            <Play className="mr-2 h-4 w-4" />
            New run
          </Link>
        </Button>
      </div>
      <p>
        A pipeline is a published research tool (a model and the code around it) that VideoAnnotator runs on your
        videos for you. Each finds one kind of thing, such as who spoke when or where faces are, and writes it as
        results you can view, compare and export. They run on the server; your videos aren&apos;t sent anywhere else.
      </p>
      <PageHelp page="pipelines">Open a pipeline to see what it finds and set your own defaults for it.</PageHelp>
      {error instanceof APIError && error.status === 0 && (
        <NeedsServer what="Pipelines are installed on a VideoAnnotator server, and run there." />
      )}

      <Tabs value={tab} onValueChange={(value) => navigate({ hash: value === 'workflows' ? 'workflows' : '' }, { replace: true })}>
        <TabsList>
          <TabsTrigger value="pipelines">Pipelines</TabsTrigger>
          <TabsTrigger value="workflows">
            Workflows{workflows.data?.length ? ` · ${workflows.data.length}` : ''}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pipelines" className="space-y-6 pt-2">
          {save.isError && (
            <p className="text-sm text-destructive" role="alert">
              Your defaults weren&apos;t saved: change one again to retry.
            </p>
          )}
          {step(others)}
          {askAI && <AskAIPanel pipeline={askAI} yours={yours} onYours={onYours} notReady={step([askAI], false)} />}
          <section className="space-y-2 text-sm" data-testid="another-tool">
            <h2 className="text-lg font-semibold">Another tool you&apos;d like?</h2>
            <p>
              VideoAnnotator comes with the pipelines above. To suggest one, say which tool and what it finds:
              <ReportProblemButton className="h-auto px-1">Suggest a tool</ReportProblemButton>
              or{' '}
              <a className="underline" href={`${GITHUB_URL}/issues/new`} target="_blank" rel="noreferrer">
                open a GitHub issue
                <ExternalLink className="ml-0.5 inline h-3 w-3" aria-hidden="true" />
              </a>
              . Many questions need no new pipeline: Ask AI looks for what you describe.
            </p>
            <p className="text-muted-foreground">
              Developers: from VideoAnnotator 1.8, a pipeline can be added as a plugin, a Python package installed beside
              it, which then shows on this page. Until then a pipeline is added to{' '}
              <a className="underline" href={`${GITHUB_URL}/tree/master/src/videoannotator/pipelines`} target="_blank" rel="noreferrer">
                VideoAnnotator&apos;s source
                <ExternalLink className="ml-0.5 inline h-3 w-3" aria-hidden="true" />
              </a>
              .
            </p>
          </section>
        </TabsContent>

        <TabsContent value="workflows" className="pt-2">
          <WorkflowsSection />
        </TabsContent>
      </Tabs>
    </div>
  );
}
