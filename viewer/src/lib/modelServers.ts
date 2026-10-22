/** Model servers in the viewer's words (VideoAnnotator spec 033 R14). */
import { DEFAULT_SERVER_ID, type ModelServer, type ModelServerKind } from '@/types/ask';

export const KIND_LABELS: Record<ModelServerKind, string> = {
  ollama: 'Ollama',
  openai_compatible: 'OpenAI-compatible (llama.cpp, vLLM, LM Studio…)',
};

export const KIND_SHORT: Record<ModelServerKind, string> = {
  ollama: 'Ollama',
  openai_compatible: 'OpenAI-compatible',
};

/** Where a server's key comes from, without ever the key. */
export function keyText(server: Pick<ModelServer, 'has_key' | 'key_source' | 'api_key_env'>): string {
  if (server.key_source === 'env' && server.api_key_env) return `key from $${server.api_key_env}`;
  if (server.has_key) return 'key set';
  if (server.api_key_env) return `no key ($${server.api_key_env} isn't set)`;
  return 'no key';
}

export const MODEL_SERVERS_QUERY_KEY = ['model-servers'] as const;

const trimmed = (url: string) => url.replace(/\/+$/, '').toLowerCase();

/** The listed server a vlm_annotation config points at, if any: empty means the default. */
export function serverForConfig(servers: ModelServer[], baseUrl: unknown): ModelServer | undefined {
  if (!baseUrl) return servers.find((s) => s.id === DEFAULT_SERVER_ID);
  return servers.find((s) => trimmed(s.base_url) === trimmed(String(baseUrl)));
}
