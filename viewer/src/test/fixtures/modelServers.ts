/** Model servers as the server lists them (spec 033), for tests. */
import type { ModelServer, ModelServersResponse } from '@/types/ask';

export const DEFAULT_SERVER: ModelServer = {
  id: 'default',
  name: 'This computer (Ollama)',
  kind: 'ollama',
  base_url: 'http://127.0.0.1:11434',
  has_key: false,
  key_source: null,
  api_key_env: null,
  endpoint: { host: '127.0.0.1', class: 'local', confirmed: true, sendable: true, before_switch: false },
  structured_output: null,
  video_models: {},
  removable: false,
};

export const LAB_SERVER: ModelServer = {
  id: 'lab',
  name: 'lab-gpu',
  kind: 'openai_compatible',
  base_url: 'http://10.0.4.12:8000/v1',
  has_key: true,
  key_source: 'stored',
  api_key_env: null,
  endpoint: { host: '10.0.4.12', class: 'local', confirmed: true, sendable: true, before_switch: false },
  structured_output: null,
  video_models: {},
  removable: true,
};

export const OLD_REMOTE_SERVER: ModelServer = {
  id: 'old',
  name: 'gpu-farm',
  kind: 'ollama',
  base_url: 'http://gpu.example.org:11434',
  has_key: false,
  key_source: null,
  api_key_env: 'FARM_KEY',
  endpoint: { host: 'gpu.example.org', class: 'remote', confirmed: true, sendable: true, before_switch: true },
  structured_output: null,
  video_models: {},
  removable: true,
};

export const REMOTE_WARNING =
  'With remote model servers switched on, frames from the videos people ask about can be sent to servers outside this computer.';

export function serversResponse(
  servers: ModelServer[] = [DEFAULT_SERVER, LAB_SERVER],
  remote: Partial<ModelServersResponse['remote']> = {},
): ModelServersResponse {
  return {
    servers,
    remote: { allowed: false, fixed_by_environment: false, warning: REMOTE_WARNING, ...remote },
    keys_file: '/data/model_server_keys.json',
  };
}

/** Radix Select needs these in jsdom. */
export function stubPointerCapture(): void {
  const proto = Element.prototype as unknown as Record<string, unknown>;
  proto.hasPointerCapture ??= () => false;
  proto.releasePointerCapture ??= () => undefined;
  proto.scrollIntoView ??= () => undefined;
}
