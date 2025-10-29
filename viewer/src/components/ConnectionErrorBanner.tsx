/**
 * Connection Error Banner
 * Displays prominent CORS/connection error guidance when server is unreachable
 */

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertCircle, Settings, ExternalLink } from 'lucide-react';
import { Link } from 'react-router-dom';

interface ConnectionErrorBannerProps {
    error: Error;
    apiUrl: string;
    onRetry?: () => void;
    className?: string;
}

/**
 * Detects if error is likely a CORS or network connectivity issue
 */
function isCorsOrNetworkError(error: Error): boolean {
    const message = error.message.toLowerCase();
    return (
        message.includes('cors') ||
        message.includes('fetch') ||
        message.includes('network') ||
        message.includes('failed to fetch') ||
        message.includes('networkerror') ||
        message.includes('access-control-allow-origin')
    );
}

export function ConnectionErrorBanner({
    error,
    apiUrl,
    onRetry,
    className,
}: ConnectionErrorBannerProps) {
    const isCorsError = isCorsOrNetworkError(error);

    return (
        <Alert variant="destructive" className={className}>
            <AlertCircle className="h-5 w-5" />
            <AlertTitle className="text-lg font-semibold">
                {isCorsError ? 'Cannot Connect to VideoAnnotator Server' : 'Server Error'}
            </AlertTitle>
            <AlertDescription className="mt-2 space-y-3">
                {isCorsError ? (
                    <>
                        <p className="text-sm">
                            The VideoAnnotator server at{' '}
                            <code className="bg-destructive/20 px-1 py-0.5 rounded text-xs">
                                {apiUrl}
                            </code>{' '}
                            is not responding or CORS is not configured correctly.
                        </p>

                        <div className="bg-destructive/10 rounded-md p-3 text-sm space-y-2">
                            <p className="font-semibold">Troubleshooting Steps:</p>
                            <ol className="list-decimal list-inside space-y-2 ml-2">
                                <li>
                                    <strong>Check the server is running</strong>
                                    <div className="ml-6 mt-1 text-xs text-muted-foreground">
                                        Open a terminal and verify VideoAnnotator is started. You should see output like "Server running on port 18011"
                                    </div>
                                </li>
                                <li>
                                    <strong>Verify the server URL is correct</strong>
                                    <div className="ml-6 mt-1 text-xs text-muted-foreground">
                                        Click "Check API Settings" below to confirm the URL matches where your server is running
                                    </div>
                                </li>
                                <li>
                                    <strong>Test the server in your browser</strong>
                                    <div className="ml-6 mt-1 text-xs text-muted-foreground">
                                        Visit{' '}
                                        <a
                                            href={`${apiUrl}/health`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="underline hover:text-destructive-foreground"
                                        >
                                            {apiUrl}/health
                                        </a>
                                        {' '}in a new tab. You should see a JSON response with server status.
                                    </div>
                                </li>
                                <li>
                                    <strong>Restart the server (if needed)</strong>
                                    <div className="ml-6 mt-1 text-xs">
                                        <p className="text-muted-foreground mb-1">For local development, the server now supports common ports automatically:</p>
                                        <code className="bg-destructive/20 px-2 py-1 rounded text-xs block mb-2">
                                            uv run videoannotator server
                                        </code>
                                        <p className="text-muted-foreground mb-1 italic">
                                            This works automatically for React, Vite, Vue, Angular on standard ports (3000, 5173, 8080, 4200, etc.)
                                        </p>
                                        <p className="text-muted-foreground mb-1 mt-2">
                                            <strong>Testing or using a custom port?</strong> Use dev mode:
                                        </p>
                                        <code className="bg-destructive/20 px-2 py-1 rounded text-xs block">
                                            uv run videoannotator server --dev
                                        </code>
                                        <p className="text-muted-foreground mt-1 italic text-xs">
                                            Dev mode allows all origins - perfect for testing!
                                        </p>
                                    </div>
                                </li>
                            </ol>
                        </div>

                        <div className="pt-2 space-y-2">
                            <p className="text-xs text-muted-foreground">
                                <strong>Still having trouble?</strong> Check the browser console (press F12, click Console tab) for more detailed error information.
                            </p>
                        </div>                        <div className="flex gap-2 pt-2">
                            <Link to="/create/settings">
                                <Button variant="outline" size="sm">
                                    <Settings className="h-4 w-4 mr-2" />
                                    Check API Settings
                                </Button>
                            </Link>
                            {onRetry && (
                                <Button onClick={onRetry} variant="outline" size="sm">
                                    Retry Connection
                                </Button>
                            )}
                            <a
                                href="https://github.com/InfantLab/VideoAnnotator#server-setup"
                                target="_blank"
                                rel="noopener noreferrer"
                            >
                                <Button variant="ghost" size="sm">
                                    <ExternalLink className="h-4 w-4 mr-2" />
                                    Server Setup Guide
                                </Button>
                            </a>
                        </div>
                    </>
                ) : (
                    <>
                        <p className="text-sm">{error.message || 'An unexpected error occurred'}</p>
                        <div className="flex gap-2 pt-2">
                            <Link to="/create/settings">
                                <Button variant="outline" size="sm">
                                    <Settings className="h-4 w-4 mr-2" />
                                    Check Settings
                                </Button>
                            </Link>
                            {onRetry && (
                                <Button onClick={onRetry} variant="outline" size="sm">
                                    Retry
                                </Button>
                            )}
                        </div>
                    </>
                )}
            </AlertDescription>
        </Alert>
    );
}
