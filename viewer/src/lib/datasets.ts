/** Helpers shared by the Datasets page, the wizard and import/export (spec 018). */
import { APIError } from '@/api/handleError';
import type { CurrentUser } from '@/types/api';

/**
 * Whether to offer edit/delete. Unknown identity (no token, older server)
 * offers them and lets the server's 403 decide, as useCurrentUser advises.
 */
export function canEdit(ownerUserId: string, user: CurrentUser | null): boolean {
  if (!user) return true;
  return user.isAdmin || String(user.id) === ownerUserId;
}

export function exportFileName(kind: 'dataset' | 'preset', name: string): string {
  const safe = name.trim().replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '') || kind;
  return `${safe}.${kind}.json`;
}

export function downloadJSON(fileName: string, data: unknown): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/**
 * Creates `body`, renaming it "<name> (imported)", "(imported 2)", … when the
 * server says the caller already has that name (409). Never overwrites.
 */
export async function createWithFreeName<B extends { name: string }, R>(
  body: B,
  create: (body: B) => Promise<R>,
): Promise<{ created: R; name: string }> {
  const base = body.name;
  for (let attempt = 0; attempt < 20; attempt++) {
    const name = attempt === 0 ? base : `${base} (imported${attempt === 1 ? '' : ` ${attempt}`})`;
    try {
      return { created: await create({ ...body, name }), name };
    } catch (error) {
      if (!(error instanceof APIError) || error.status !== 409) throw error;
    }
  }
  throw new Error(`Couldn't find a free name for "${base}"; rename it and try again.`);
}

/** The parts of an exported definition the server accepts back; anything else is a clear error. */
export function parseExport(text: string, required: string[]): Record<string, unknown> {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("This isn't a JSON file.");
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error("This file doesn't contain an exported definition.");
  }
  const missing = required.filter((field) => !(field in (data as Record<string, unknown>)));
  if (missing.length > 0) {
    throw new Error(`This file is missing ${missing.join(', ')}; is it an export from VideoAnnotator?`);
  }
  return data as Record<string, unknown>;
}
