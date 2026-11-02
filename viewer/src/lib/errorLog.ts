/**
 * The viewer's own recent failures, kept in memory for a problem report
 * (UAT #68). Not the browser console: only what the viewer itself hit.
 */
export interface LoggedError {
  at: Date;
  message: string;
}

const MAX = 20;
const log: LoggedError[] = [];

/** Folder paths and keys out of a message: a report never carries them. */
export function scrub(message: string): string {
  return message
    .replace(/va_[A-Za-z0-9_-]{20,}/g, 'va_…')
    .replace(/([?&](?:token|key)=)[^&\s]+/gi, '$1…')
    .replace(/(?:[A-Za-z]:)?(?:[\\/][^\\/\s"'`]+)+[\\/]([^\\/\s"'`]+)/g, '…/$1');
}

export function recordError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  if (!message) return;
  const entry = { at: new Date(), message: scrub(message).slice(0, 300) };
  // A message shown on every render is one error, not dozens.
  if (log.length && log[log.length - 1].message === entry.message) log[log.length - 1] = entry;
  else log.push(entry);
  if (log.length > MAX) log.splice(0, log.length - MAX);
}

export function recentErrors(count = 3): LoggedError[] {
  return log.slice(-count).reverse();
}

export function clearErrors(): void {
  log.length = 0;
}

let installed = false;

/** Uncaught errors and rejections from the viewer's own code. */
export function installErrorLog(): void {
  if (installed || typeof window === 'undefined') return;
  installed = true;
  window.addEventListener('error', (event) => recordError(event.error ?? event.message));
  window.addEventListener('unhandledrejection', (event) => recordError(event.reason));
}
