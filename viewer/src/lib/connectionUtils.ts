/**
 * Shared connection error detection utilities
 *
 * Detects if an error is likely a CORS or network connectivity issue.
 * Auth errors (401, 403, 404) are NOT treated as connection issues.
 */

export function isCorsOrNetworkError(error: Error): boolean {
  const message = error.message.toLowerCase();

  // Auth/permission errors should NOT show the connection banner
  const isAuthError =
    message.includes('401') ||
    message.includes('403') ||
    message.includes('404') ||
    message.includes('unauthorized') ||
    message.includes('forbidden') ||
    message.includes('not found') ||
    message.includes('auth') ||
    message.includes('permission');

  if (isAuthError) return false;

  // Actual connection issues
  return (
    message.includes('cors') ||
    message.includes('fetch') ||
    message.includes('network') ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('timeout') ||
    message.includes('timed out') ||
    message.includes('access-control-allow-origin')
  );
}
