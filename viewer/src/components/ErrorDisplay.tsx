/**
 * T047-T049: ErrorDisplay Component
 * Displays errors with message, hint, collapsible technical details, and field errors
 */

import { useState } from 'react';
import { AlertCircle, ChevronDown, ChevronRight, Copy, CheckCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import type { ParsedError } from '@/types/api';

interface ErrorDisplayProps {
  error: ParsedError;
  className?: string;
}

export function ErrorDisplay({ error, className }: ErrorDisplayProps) {
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const hasTechnicalDetails = error.code || error.requestId;

  const handleCopy = async () => {
    const technicalInfo = [
      error.message,
      error.code && `Error Code: ${error.code}`,
      error.requestId && `Request ID: ${error.requestId}`,
      error.hint && `Hint: ${error.hint}`,
    ]
      .filter(Boolean)
      .join('\n');

    try {
      await navigator.clipboard.writeText(technicalInfo);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <Alert variant="destructive" className={className} role="alert">
      <div className="flex items-start gap-3">
        <AlertCircle className="h-5 w-5 mt-0.5 flex-shrink-0" data-testid="error-icon" />
        
        <div className="flex-1 space-y-3">
          {/* Main Error Message */}
          <div>
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              <p className="text-sm">{error.message}</p>
            </AlertDescription>
          </div>

          {/* Hint */}
          {error.hint && (
            <div className="bg-destructive/10 rounded-md p-3 border-l-4 border-destructive/30">
              <p className="text-sm font-medium">💡 Tip:</p>
              <p className="text-sm mt-1">{error.hint}</p>
            </div>
          )}

          {/* Field Errors */}
          {error.fieldErrors && error.fieldErrors.length > 0 && (
            <div className="space-y-2" data-testid="field-errors">
              <p className="text-sm font-medium">Issues found:</p>
              <ul className="space-y-2">
                {error.fieldErrors.map((fieldError, index) => (
                  <li
                    key={index}
                    className="bg-destructive/10 rounded-md p-2 border-l-2 border-destructive"
                    data-testid="field-error-item"
                  >
                    <p className="text-sm">
                      <span className="font-mono font-medium">{fieldError.field}:</span>{' '}
                      {fieldError.message}
                    </p>
                    {fieldError.hint && (
                      <p className="text-xs mt-1 text-muted-foreground">
                        💡 {fieldError.hint}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Technical Details (Collapsible) */}
          {hasTechnicalDetails && (
            <div className="border-t pt-3">
              <button
                onClick={() => setDetailsExpanded(!detailsExpanded)}
                className="flex items-center gap-2 text-sm font-medium hover:underline"
                aria-expanded={detailsExpanded}
                aria-controls="technical-details"
              >
                {detailsExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
                Technical Details
              </button>

              {detailsExpanded && (
                <div
                  id="technical-details"
                  className="mt-3 space-y-2 bg-muted/50 rounded-md p-3"
                >
                  {error.code && (
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="text-xs font-medium text-muted-foreground">
                          Error Code:
                        </p>
                        <p className="text-sm font-mono">{error.code}</p>
                      </div>
                    </div>
                  )}

                  {error.requestId && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">
                        Request ID:
                      </p>
                      <p className="text-sm font-mono break-all">{error.requestId}</p>
                    </div>
                  )}

                  <div className="pt-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleCopy}
                      className="h-8"
                    >
                      {copied ? (
                        <>
                          <CheckCircle className="h-3 w-3 mr-2" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="h-3 w-3 mr-2" />
                          Copy Details
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Alert>
  );
}
