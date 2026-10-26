/**
 * @file Defines a custom error class and utility function for handling API errors.
 */

/**
 * Custom error class for API-specific errors.
 * This allows distinguishing between network errors, HTTP errors, and other exceptions.
 */
/** Shown for every 401, whatever the server said (UAT #39: never "set AUTH_REQUIRED=false"). */
export const NEEDS_KEY_MESSAGE =
    'This server needs your key. Open the link you were given, or paste the key in Settings.';

export class APIError extends Error {
    /** What the server said, kept for technical details when `message` is replaced. */
    public serverMessage: string;

    /**
     * @param message - The error message.
     * @param status - The HTTP status code of the response (0 for network errors).
     * @param response - The original fetch Response object, if available.
     * @param body - The parsed JSON error body, if there was one. The response
     *   stream is already consumed, so this is the only way to read the error
     *   envelope's `code`/`hint`/`details` (see `apiErrorEnvelope`).
     */
    constructor(
        message: string,
        public status: number,
        public response?: Response,
        public body?: unknown
    ) {
        super(status === 401 ? NEEDS_KEY_MESSAGE : message);
        this.name = 'APIError';
        this.serverMessage = message;
    }
}

/** The machine-readable parts of VideoAnnotator's `{ error: { code, hint, details } }` envelope. */
export interface APIErrorEnvelope {
    code?: string;
    hint?: string;
    details?: Record<string, unknown>;
}

export const isNeedsKey = (error: unknown): boolean => error instanceof APIError && error.status === 401;

export const apiErrorEnvelope = (error: unknown): APIErrorEnvelope => {
    if (!(error instanceof APIError) || !error.body || typeof error.body !== 'object') return {};
    const envelope = (error.body as Record<string, unknown>).error;
    if (!envelope || typeof envelope !== 'object') return {};
    const { code, hint, details } = envelope as Record<string, unknown>;
    return {
        code: typeof code === 'string' ? code : undefined,
        hint: typeof hint === 'string' ? hint : undefined,
        details: details && typeof details === 'object' ? (details as Record<string, unknown>) : undefined
    };
};

/**
 * A helper function to convert an unknown error into a user-friendly string.
 * @param error - The error to handle, which can be of any type.
 * @returns A string message suitable for display in the UI.
 */
export const handleAPIError = (error: unknown): string => {
    if (error instanceof APIError) {
        return error.message;
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'An unexpected error occurred';
};
