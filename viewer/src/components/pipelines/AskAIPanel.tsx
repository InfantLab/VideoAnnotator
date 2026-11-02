import { useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight, Sparkles } from 'lucide-react';

import { PipelineSettings } from '@/components/DynamicPipelineParameters';
import { Button } from '@/components/ui/button';
import { usePipelineSchema } from '@/hooks/usePipelineCatalog';
import { useVlmModels } from '@/hooks/useVlmModels';
import { isPipelineSelectable } from '@/lib/pipelineExtras';
import { kindColour, RESULT_KINDS } from '@/lib/resultKinds';
import type { PipelineDescriptor } from '@/types/pipelines';

type Updater = (prev: Record<string, unknown>) => Record<string, unknown>;

/** Whether a model server answers, as checked now (P1). */
function ModelLine() {
  const models = useVlmModels();
  if (models.isLoading) return <>Checking the model server…</>;
  if (models.error || !models.data) {
    return (
      <span className="text-amber-700 dark:text-amber-400" data-testid="ask-model-line">
        No model server answers yet: add one under Model servers.
      </span>
    );
  }
  const { kind, baseUrl, models: names } = models.data;
  const which = kind === 'openai_compatible' ? 'A model server' : 'Ollama';
  return (
    <span data-testid="ask-model-line">
      Now: {which} at {baseUrl} answers
      {names.length ? `, with ${names.length === 1 ? names[0] : `${names.length} models`}` : ', but has no models yet'}.
    </span>
  );
}

/**
 * Ask AI, set apart from the other pipelines (UAT #71): it is a question you
 * write to a model you choose, not one published model finding one thing.
 */
export function AskAIPanel({
  pipeline,
  yours,
  onYours,
  notReady,
}: {
  pipeline: PipelineDescriptor;
  yours: Record<string, unknown>;
  onYours: (updater: Updater) => void;
  /** Its install or setup card, when it can't run yet. */
  notReady: ReactNode;
}) {
  const ready = isPipelineSelectable(pipeline);
  const [open, setOpen] = useState(false);
  const schema = usePipelineSchema(pipeline.id, { enabled: open });
  const colour = kindColour(RESULT_KINDS.ask_ai);

  return (
    <section className="space-y-3 rounded-lg border-2 p-4" style={{ borderColor: colour }} data-testid="ask-ai-panel">
      <h2 className="flex items-center gap-2 text-lg font-semibold">
        <Sparkles className="h-5 w-5" style={{ color: colour }} aria-hidden="true" />
        Ask AI
      </h2>
      <p className="text-sm">
        Different from the pipelines above. Each of those is one published model, tested on a benchmark, finding one
        fixed thing. With Ask AI you write the question, and you choose the model that answers it.
      </p>
      <ul className="list-disc space-y-1 pl-5 text-sm">
        <li>
          Your question decides what&apos;s found: &ldquo;Is the child looking at the caregiver?&rdquo; Answers are as
          good as the question and the model.
        </li>
        <li>No benchmark applies. Try a question on a few moments first, check the answers yourself, then run it on your videos.</li>
        <li>
          The model runs on a model server you set up (Ollama, LM Studio, llama.cpp…), on this computer or another.{' '}
          {ready && <ModelLine />}
        </li>
        <li>Slower than the others: each moment is one question to the model.</li>
      </ul>
      {!ready && notReady}
      <div className="flex flex-wrap items-center gap-2">
        <Button asChild size="sm">
          <Link to="/ask">Try a question</Link>
        </Button>
        <Button asChild size="sm" variant="outline">
          <Link to="/settings#ask">Model servers</Link>
        </Button>
        {ready && (
          <button
            type="button"
            className="ml-auto flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            Your defaults
            <ChevronRight className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-90' : ''}`} aria-hidden="true" />
          </button>
        )}
      </div>
      {open && schema.data && (
        <div className="divide-y border-t">
          <PipelineSettings pipeline={{ ...pipeline, parameters: schema.data.parameters }} config={yours} onConfigChange={onYours} describe />
        </div>
      )}
    </section>
  );
}
