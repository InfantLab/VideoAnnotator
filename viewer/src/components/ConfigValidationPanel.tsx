// ConfigValidationPanel component
// Displays configuration validation results with errors, warnings, and hints

import React, { useState } from 'react';
import type { ConfigValidationResult } from '@/types/api';
import type { ValidationIssue } from '@/types/api';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { ChevronDown, ChevronRight, CheckCircle2, AlertCircle, AlertTriangle, Loader2 } from 'lucide-react';

interface ConfigValidationPanelProps {
    validationResult: ConfigValidationResult | null | undefined;
    isValidating: boolean;
    className?: string;
    devMode?: boolean; // Optional: override environment dev mode
}

/**
 * Displays configuration validation results
 * Groups errors/warnings by field, shows hints and suggestions
 */
export function ConfigValidationPanel({
    validationResult,
    isValidating,
    className = '',
    devMode,
}: ConfigValidationPanelProps) {
    const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
    const isDev = devMode ?? import.meta.env.DEV;

    // Update expanded sections when validationResult changes
    React.useEffect(() => {
        if (validationResult) {
            const issuesByField = new Map<string, ValidationIssue[]>();
            [...(validationResult.errors || []), ...(validationResult.warnings || [])].forEach((issue) => {
                const fullField = issue.field || 'general';
                const parentField = fullField.split('.')[0];
                if (!issuesByField.has(parentField)) {
                    issuesByField.set(parentField, []);
                }
                issuesByField.get(parentField)!.push(issue);
            });

            // Expand all fields by default
            setExpandedSections(new Set(Array.from(issuesByField.keys())));
        }
    }, [validationResult]);

    // Don't render if no validation result and not validating
    if (!validationResult && !isValidating) {
        return null;
    }

    // Show validating indicator
    if (isValidating) {
        return (
            <div className={`flex items-center gap-2 text-muted-foreground ${className}`}>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Validating configuration...</span>
            </div>
        );
    }

    // No result yet
    if (!validationResult) {
        return null;
    }

    const { valid, errors = [], warnings = [] } = validationResult;

    // Show success message for valid config with no warnings
    if (valid && errors.length === 0 && warnings.length === 0) {
        return (
            <Alert className={`border-green-200 bg-green-50 ${className}`}>
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertTitle className="text-green-900">Configuration Valid</AlertTitle>
                <AlertDescription className="text-green-700">
                    Your configuration is valid and ready to use.
                </AlertDescription>
            </Alert>
        );
    }

    // Group issues by parent field (first segment before dot)
    const issuesByField = new Map<string, typeof errors>();
    [...errors, ...warnings].forEach((issue) => {
        const fullField = issue.field || 'general';
        // Extract parent field (e.g., "scene_detection.threshold" -> "scene_detection")
        const parentField = fullField.split('.')[0];

        if (!issuesByField.has(parentField)) {
            issuesByField.set(parentField, []);
        }
        issuesByField.get(parentField)!.push(issue);
    });

    const toggleSection = (field: string) => {
        const newExpanded = new Set(expandedSections);
        if (newExpanded.has(field)) {
            newExpanded.delete(field);
        } else {
            newExpanded.add(field);
        }
        setExpandedSections(newExpanded);
    };

    return (
        <div className={`space-y-4 ${className}`} role="region" aria-label="Configuration validation results">
            {/* Summary */}
            {errors.length > 0 && (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Configuration Errors</AlertTitle>
                    <AlertDescription>
                        {errors.length} error{errors.length !== 1 ? 's' : ''} found. Please fix these before submitting.
                    </AlertDescription>
                </Alert>
            )}

            {warnings.length > 0 && errors.length === 0 && (
                <Alert className="border-yellow-200 bg-yellow-50">
                    <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    <AlertTitle className="text-yellow-900">Configuration Warnings</AlertTitle>
                    <AlertDescription className="text-yellow-700">
                        {warnings.length} warning{warnings.length !== 1 ? 's' : ''} found. You can proceed, but review these suggestions.
                    </AlertDescription>
                </Alert>
            )}

            {/* Issues grouped by field */}
            <div className="space-y-2">
                {Array.from(issuesByField.entries()).map(([field, issues]) => {
                    const isExpanded = expandedSections.has(field);
                    const hasErrors = issues.some(issue => issue.severity === 'error');
                    const hasWarnings = issues.some(issue => issue.severity === 'warning');

                    return (
                        <Collapsible
                            key={field}
                            open={isExpanded}
                            onOpenChange={() => toggleSection(field)}
                            className="border rounded-lg p-4 bg-card"
                        >
                            <CollapsibleTrigger className="flex items-center justify-between w-full text-left">
                                <div className="flex items-center gap-2">
                                    {isExpanded ? (
                                        <ChevronDown className="h-4 w-4" />
                                    ) : (
                                        <ChevronRight className="h-4 w-4" />
                                    )}
                                    <span className="font-medium">{field}</span>
                                    <div className="flex gap-1">
                                        {hasErrors && (
                                            <Badge variant="destructive" className="text-xs">
                                                {issues.filter(i => i.severity === 'error').length} error{issues.filter(i => i.severity === 'error').length !== 1 ? 's' : ''}
                                            </Badge>
                                        )}
                                        {hasWarnings && (
                                            <Badge variant="outline" className="text-xs border-yellow-400 text-yellow-700">
                                                {issues.filter(i => i.severity === 'warning').length} warning{issues.filter(i => i.severity === 'warning').length !== 1 ? 's' : ''}
                                            </Badge>
                                        )}
                                    </div>
                                </div>
                            </CollapsibleTrigger>

                            <CollapsibleContent className="mt-3 space-y-3">
                                {issues.map((issue, index) => (
                                    <div
                                        key={`${field}-${index}`}
                                        className={`p-3 rounded-md ${issue.severity === 'error'
                                            ? 'bg-red-50 border border-red-200'
                                            : 'bg-yellow-50 border border-yellow-200'
                                            }`}
                                    >
                                        <div className="flex items-start gap-2">
                                            {issue.severity === 'error' ? (
                                                <AlertCircle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
                                            ) : (
                                                <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                                            )}
                                            <div className="flex-1 space-y-1">
                                                {/* Show sub-field name if field has a dot */}
                                                {issue.field && issue.field.includes('.') && (
                                                    <p className={`text-xs font-mono ${issue.severity === 'error' ? 'text-red-600' : 'text-yellow-600'
                                                        }`}>
                                                        {issue.field.split('.').slice(1).join('.')}
                                                    </p>
                                                )}

                                                <p className={`text-sm font-medium ${issue.severity === 'error' ? 'text-red-900' : 'text-yellow-900'
                                                    }`}>
                                                    {issue.message}
                                                </p>

                                                {issue.hint && (
                                                    <p className={`text-sm ${issue.severity === 'error' ? 'text-red-700' : 'text-yellow-700'
                                                        }`}>
                                                        <span className="font-medium">Hint:</span> {issue.hint}
                                                    </p>
                                                )}

                                                {issue.suggested_value !== undefined && (
                                                    <p className={`text-sm ${issue.severity === 'error' ? 'text-red-700' : 'text-yellow-700'
                                                        }`}>
                                                        <span className="font-medium">Suggested:</span>{' '}
                                                        <code className="px-1 py-0.5 rounded bg-black/10">
                                                            {JSON.stringify(issue.suggested_value)}
                                                        </code>
                                                    </p>
                                                )}

                                                {isDev && issue.error_code && (
                                                    <p className={`text-xs font-mono ${issue.severity === 'error' ? 'text-red-600' : 'text-yellow-600'
                                                        }`}>
                                                        Code: {issue.error_code}
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </CollapsibleContent>
                        </Collapsible>
                    );
                })}
            </div>
        </div>
    );
}
