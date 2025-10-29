/**
 * T052: ErrorBoundary component
 * Catches React rendering errors and displays them with ErrorDisplay
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { parseApiError } from '@/lib/errorHandling';
import { Button } from '@/components/ui/button';
import { RefreshCw } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: (error: Error, resetError: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary component that catches React rendering errors
 * and displays them using ErrorDisplay component
 * 
 * Usage:
 * ```tsx
 * <ErrorBoundary>
 *   <YourComponent />
 * </ErrorBoundary>
 * ```
 * 
 * With custom fallback:
 * ```tsx
 * <ErrorBoundary fallback={(error, reset) => (
 *   <div>
 *     <h1>Oops!</h1>
 *     <p>{error.message}</p>
 *     <button onClick={reset}>Try again</button>
 *   </div>
 * )}>
 *   <YourComponent />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console for debugging
    console.error('React Error Boundary caught an error:', error, errorInfo);
    
    // You can also log to an error reporting service here
    // Example: logErrorToService(error, errorInfo);
  }

  resetError = (): void => {
    this.setState({
      hasError: false,
      error: null,
    });
  };

  render(): ReactNode {
    if (this.state.hasError && this.state.error) {
      // If custom fallback provided, use it
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.resetError);
      }

      // Default fallback UI with ErrorDisplay
      const parsedError = parseApiError({
        error: this.state.error.message,
        error_code: 'REACT_ERROR',
        hint: 'Try refreshing the page or clearing your browser cache. If the problem persists, please report this issue.',
      });

      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-muted/30">
          <div className="max-w-2xl w-full space-y-4">
            <div className="text-center space-y-2">
              <h1 className="text-3xl font-bold">Something went wrong</h1>
              <p className="text-muted-foreground">
                The application encountered an unexpected error
              </p>
            </div>

            <ErrorDisplay error={parsedError} />

            <div className="flex gap-3 justify-center">
              <Button onClick={this.resetError} variant="default">
                <RefreshCw className="h-4 w-4 mr-2" />
                Try Again
              </Button>
              <Button onClick={() => window.location.href = '/'} variant="outline">
                Go Home
              </Button>
            </div>

            {/* Stack trace in development */}
            {process.env.NODE_ENV === 'development' && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
                  Stack Trace (Development Only)
                </summary>
                <pre className="mt-2 text-xs bg-muted p-4 rounded overflow-x-auto">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
