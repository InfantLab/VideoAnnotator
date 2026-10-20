import { Link } from 'react-router-dom';
import { PageHelp } from '@/components/PageHelp';
import { WorkflowsSection } from '@/components/workflows/WorkflowsSection';
import { Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePipelineCatalog, useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';
import { PipelineSelectionStep } from '@/components/run/PipelineChoice';

/**
 * Installing a pipeline is a one-off setup task, so it gets its own page; the
 * New run starts from videos, which would hide installing from newcomers.
 */
export default function Pipelines() {
  const { data, isLoading, error } = usePipelineCatalog();
  const refresh = useRefreshPipelineCatalog();

  return (
    <div className="container mx-auto px-6 py-8 max-w-4xl space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <h1 className="text-3xl font-bold">Pipelines</h1>
        <Button asChild>
          <Link to="/run">
            <Play className="h-4 w-4 mr-2" />
            Run pipelines on videos
          </Link>
        </Button>
      </div>
      <PageHelp page="pipelines">What each pipeline finds, and installing the ones you need. You choose which to run when you start a run.</PageHelp>
      <PipelineSelectionStep
        manage
        pipelines={data?.catalog.pipelines ?? []}
        selectedPipelines={[]}
        setSelectedPipelines={() => {}}
        isLoading={isLoading}
        error={error}
        onRetry={() => refresh({ forceServerRefresh: true })}
        restartRequired={data?.restartRequired ?? false}
      />
      <WorkflowsSection />
    </div>
  );
}
