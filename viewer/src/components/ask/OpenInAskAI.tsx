import { Sparkles } from 'lucide-react';
import { samplingFromConfig } from '@/lib/sampling';

interface OpenInAskAIProps {
  config: Record<string, unknown>;
}

/**
 * Where the wizard's prompt test panel was (spec 027 FR-004): the question,
 * model and sampling, in Ask AI. A new tab keeps the wizard as it is; router
 * state doesn't cross tabs, the URL does.
 */
export function OpenInAskAI({ config }: OpenInAskAIProps) {
  const prompt = typeof config.prompt === 'string' ? config.prompt : '';
  const model = typeof config.model === 'string' ? config.model : '';
  const query = new URLSearchParams({
    ...(prompt.trim() ? { prompt } : {}),
    ...(model ? { model } : {}),
    sampling: JSON.stringify(samplingFromConfig(config)),
  });
  return (
    <a
      href={`${import.meta.env.BASE_URL}ask?${query}`}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1.5 text-sm underline"
      data-testid="open-in-ask-ai"
    >
      <Sparkles className="h-3.5 w-3.5" /> Try this question in Ask AI
    </a>
  );
}
