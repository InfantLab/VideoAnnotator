/**
 * @file Defines a custom error class and utility function for handling API errors.
 */

/**
 * Custom error class for API-specific errors.
 * This allows distinguishing between network errors, HTTP errors, and other exceptions.
 */
export class APIError extends Error {
    /**
     * @param message - The error message.
     * @param status - The HTTP status code of the response (0 for network errors).
     * @param response - The original fetch Response object, if available.
     */
    constructor(
        message: string,
        public status: number,
        public response?: Response
    ) {
        super(message);
        this.name = 'APIError';
    }
}

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
