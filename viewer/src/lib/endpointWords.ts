/** How the Ask AI badge says where the model runs (spec 027 FR-014). */
import type { ModelEndpoint } from '@/types/ask';

const THIS_COMPUTER = new Set(['localhost', 'host.docker.internal', '::1', '[::1]']);

/**
 * Whether a local host is this computer itself rather than another on the
 * network: wording only. Whether the endpoint is local at all is the
 * server's call (`endpoint.class`), never the browser's.
 */
export const isThisComputer = (host: string): boolean => THIS_COMPUTER.has(host) || /^127\./.test(host);

export function badgeText(endpoint: ModelEndpoint, model: string): string {
  if (endpoint.class === 'remote') return `Remote · frames are sent to ${endpoint.host}`;
  const what = model || 'the model';
  return isThisComputer(endpoint.host) ? `Local · ${what} on this computer` : `Local · ${what} on ${endpoint.host}`;
}

export function localNote(endpoint: ModelEndpoint): string {
  return isThisComputer(endpoint.host)
    ? 'Frames from your video are sent only to this computer.'
    : `Frames from your video are sent only to ${endpoint.host}, on your local network.`;
}

export const NOTE_DISMISSED_KEY = 'videoannotator_ask_local_note_seen';
