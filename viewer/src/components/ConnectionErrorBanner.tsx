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
        message.includes('timeout') ||
        message.includes('timed out') ||
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
                            {error.message.toLowerCase().includes('access-control-allow-origin') 
                                ? 'is blocking requests from this web app (CORS issue).'
                                : 'is not responding or cannot be reached.'}
                        </p>



                        <div className="bg-destructive/10 rounded-md p-3 text-sm space-y-2">
                            <p className="font-semibold">Troubleshooting Steps:</p>
                            <ol className="list-decimal list-inside space-y-2 ml-2">
                                <li>
                                    <strong>Stop your current server</strong>
                                    <div className="ml-6 mt-1 text-xs text-muted-foreground">
                                        If VideoAnnotator is already running, stop it (Ctrl+C in the terminal)
                                    </div>
                                </li>
                                <li>
                                    <strong>Start the VideoAnnotator server</strong>
                                    <div className="ml-6 mt-1 space-y-2">
                                        <div className="bg-green-50 dark:bg-green-900/20 border-2 border-green-600 dark:border-green-500 rounded p-2">
                                            <p className="text-green-900 dark:text-green-200 mb-1 font-semibold">✅ Simple command (auto-configured for port 19011)</p>
                                            <code className="bg-green-100 dark:bg-green-900 px-2 py-1 rounded text-xs block font-mono">
                                                uv run videoannotator
                                            </code>
                                            <p className="text-green-800 dark:text-green-300 text-xs mt-1.5">
                                                Server automatically allows requests from port 19011 (this web app)
                                            </p>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <strong>Verify connection works</strong>
                                    <div className="ml-6 mt-1 text-xs text-muted-foreground">
                                        After restarting the server, click "Retry Connection" below
                                    </div>
                                </li>
                            </ol>
                        </div>

                        <div className="pt-2 space-y-2">
                            <p className="text-xs text-muted-foreground">
                                <strong>Note:</strong> As of VideoAnnotator v1.3.0+, port 19011 (this web app) is automatically whitelisted. Just start the server with <code className="bg-muted px-1 rounded">uv run videoannotator</code>.
                            </p>
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
