// Where the viewer finds the API, and what a key looks like.
//
// An empty API URL means "the origin that served this page": the server itself
// when it serves the viewer at /viewer, or Vite's proxy in development. The
// browser keeps localStorage (and so the saved key) per origin, and treats
// localhost and 127.0.0.1 as different origins, so an API URL that names this
// page's own origin is kept relative rather than rewritten to another host.

export const API_URL_STORAGE_KEY = 'videoannotator_api_url';
export const API_TOKEN_STORAGE_KEY = 'videoannotator_api_token';

/** Where a standalone viewer (not served by a VideoAnnotator server) looks first. */
export const STANDALONE_API_URL = 'http://127.0.0.1:18011';

/** `va_` + secrets.token_urlsafe(32), as the server's APIKeyCRUD.create makes them. */
export const API_KEY_PATTERN = /^va_[A-Za-z0-9_-]{43}$/;

export const GENERATE_TOKEN_COMMAND = 'videoannotator generate-token';

/** True when the page is served from the API's own origin (the server's /viewer, or Vite's proxy). */
export function servedWithApi(): boolean {
    return import.meta.env.MODE === 'embedded' || import.meta.env.DEV;
}

export function defaultApiUrl(): string {
    const fromEnv = import.meta.env.VITE_API_BASE_URL;
    if (fromEnv) return normalizeApiUrl(fromEnv);
    return servedWithApi() ? '' : STANDALONE_API_URL;
}

/**
 * The URL the API client should call. This page's own origin becomes '' (relative);
 * another `localhost` server becomes 127.0.0.1, because the server binds IPv4 and
 * `localhost` can resolve to ::1 first.
 */
export function normalizeApiUrl(url: string): string {
    const trimmed = url.trim().replace(/\/+$/, '');
    if (!trimmed) return '';
    let parsed: URL;
    try {
        parsed = new URL(trimmed);
    } catch {
        return trimmed;
    }
    if (typeof window !== 'undefined' && parsed.origin === window.location.origin && parsed.pathname === '/') {
        return '';
    }
    if (parsed.hostname === 'localhost') {
        parsed.hostname = '127.0.0.1';
        return parsed.toString().replace(/\/+$/, '');
    }
    return trimmed;
}

/** What the user sees for an API URL: '' is shown as this page's origin. */
export function describeApiUrl(url: string): string {
    return url || (typeof window !== 'undefined' ? window.location.origin : '');
}

/**
 * Why a pasted key can't be right, before it's sent anywhere; null if it looks fine.
 * JWTs (from the login endpoint) are accepted as they are.
 */
export function tokenFormatProblem(token: string): string | null {
    const value = token.trim();
    if (!value || value.startsWith('eyJ') || API_KEY_PATTERN.test(value)) return null;
    if (/^bearer\s/i.test(value)) {
        return 'Paste only the key, without "Bearer ".';
    }
    if (value === 'dev-token' || value === 'test-token') {
        return `"${value}" was a placeholder in older versions; the server no longer accepts it.`;
    }
    if (value.startsWith('va_')) {
        return `A key is "va_" followed by 43 characters; this one has ${value.length - 3}. Check it was copied whole.`;
    }
    return 'A key starts with "va_" followed by 43 characters.';
}

export type ConnectionCheck =
    | { ok: true; user: { name: string; isAdmin: boolean } | null; authRequired: boolean }
    | { ok: false; problem: 'no-server' | 'needs-key' | 'key-refused' | 'server-error'; authRequired?: boolean };

/**
 * Whether `token` works on the server at `url`, checked with exactly what was
 * typed (UAT #51): the saved key plays no part. Health says whether a key is
 * needed; /auth/me says whose the key is.
 */
export async function checkConnection(url: string, token: string): Promise<ConnectionCheck> {
    const base = normalizeApiUrl(url);
    const key = token.trim();
    let authRequired = false;
    try {
        const health = await fetch(`${base}/api/v1/system/health`);
        if (!health.ok) return { ok: false, problem: 'server-error' };
        const body = await health.json().catch(() => null);
        authRequired = body?.security?.auth_required === true;
    } catch {
        return { ok: false, problem: 'no-server' };
    }
    if (!key) return authRequired ? { ok: false, problem: 'needs-key', authRequired } : { ok: true, user: null, authRequired };
    try {
        const me = await fetch(`${base}/api/v1/auth/me`, { headers: { Authorization: `Bearer ${key}` } });
        if (me.status === 401 || me.status === 403) return { ok: false, problem: 'key-refused', authRequired };
        if (me.ok) {
            const user = await me.json();
            return {
                ok: true,
                user: { name: user.username || user.email || 'your key', isAdmin: user.is_admin === true },
                authRequired,
            };
        }
        // A server from before /auth/me: a list it guards says whether the key works.
        const jobs = await fetch(`${base}/api/v1/jobs?per_page=1`, { headers: { Authorization: `Bearer ${key}` } });
        if (jobs.ok) return { ok: true, user: null, authRequired };
        if (jobs.status === 401 || jobs.status === 403 || jobs.status === 404) {
            return { ok: false, problem: 'key-refused', authRequired };
        }
        return { ok: false, problem: 'server-error', authRequired };
    } catch {
        return { ok: false, problem: 'no-server' };
    }
}
