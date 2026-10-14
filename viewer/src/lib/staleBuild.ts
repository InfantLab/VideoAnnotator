// A tab opened before VideoAnnotator was upgraded (or the viewer rebuilt)
// still runs the old build, whose lazily loaded pages are hashed files the
// server no longer has. Loading one fails; a reload picks up the new build.

const RELOAD_KEY = 'videoannotator_stale_build_reload_at';
// One automatic reload per this window, so a genuinely missing file can't
// make the page reload forever.
const RELOAD_GUARD_MS = 10_000;

/** Browser messages for a lazily loaded module that couldn't be fetched. */
export function isStaleBuildError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '');
  return /dynamically imported module|Importing a module script failed|module script failed|Unable to preload CSS/i.test(
    message
  );
}

/** Reload to get the current build, unless we just did. Returns whether it reloaded. */
export function reloadForNewBuild(): boolean {
  try {
    const last = Number(sessionStorage.getItem(RELOAD_KEY) ?? 0);
    if (Date.now() - last < RELOAD_GUARD_MS) return false;
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()));
  } catch {
    // No sessionStorage (private mode): reload anyway; the browser's own
    // error screen is the worst case, as before.
  }
  window.location.reload();
  return true;
}
