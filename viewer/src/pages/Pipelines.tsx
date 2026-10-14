import { Link } from 'react-router-dom';
import { Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePipelineCatalog, useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';
import { PipelineSelectionStep } from './NewJob';

/**
 * Installing a pipeline is a one-off setup task, so it gets its own page; the
 * New job wizard asks for videos before pipelines, which hid it from newcomers.
 */
export default function Pipelines() {
  const { data, isLoading, error } = usePipelineCatalog();
  const refresh = useRefreshPipelineCatalog();

  return (
    <div className="container mx-auto px-6 py-8 max-w-4xl space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <h1 className="text-3xl font-bold">Pipelines</h1>
        <Button asChild>
          <Link to="/jobs/new">
            <Play className="h-4 w-4 mr-2" />
            Run pipelines on videos
          </Link>
        </Button>
      </div>
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
    </div>
  );
}
