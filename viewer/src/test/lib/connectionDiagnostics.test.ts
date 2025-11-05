/**
 * Tests for connection diagnostics utility
 */

import { describe, it, expect } from 'vitest';
import { formatDiagnosticReport, type DiagnosticReport } from '@/lib/connectionDiagnostics';

describe('connectionDiagnostics', () => {
    // Note: runConnectionDiagnostics is complex to test in isolation due to:
    // - Multiple sequential fetch calls with branching logic
    // - Nested try-catch blocks that depend on previous test results
    // - Browser-specific behavior (DNS resolution, CORS)
    // 
    // Integration testing via Playwright is more appropriate for the full diagnostic flow.
    // Here we test the report formatting which is pure logic.

    describe('formatDiagnosticReport', () => {
        it('should format a successful report', () => {
            const report = {
                timestamp: new Date('2025-01-01T12:00:00Z').toISOString(),
                summary: 'all_passed' as const,
                results: [
                    {
                        test: 'Server Address',
                        passed: true,
                        duration: 123
                    },
                    {
                        test: 'Server Reachable',
                        passed: true,
                        duration: 456
                    }
                ],
                recommendations: []
            };

            const formatted = formatDiagnosticReport(report);

            expect(formatted).toContain('Connection Diagnostics Report');
            expect(formatted).toContain('ALL PASSED');
            expect(formatted).toContain('✅ PASS Server Address');
            expect(formatted).toContain('123ms');
        });

        it('should format a failed report with recommendations', () => {
            const report = {
                timestamp: new Date('2025-01-01T12:00:00Z').toISOString(),
                summary: 'partial_failure' as const,
                results: [
                    {
                        test: 'API Token Valid',
                        passed: false,
                        error: 'Invalid token'
                    }
                ],
                recommendations: [
                    '🔑 Your API token is invalid',
                    '💡 Get a new token from the administrator'
                ]
            };

            const formatted = formatDiagnosticReport(report);

            expect(formatted).toContain('PARTIAL FAILURE');
            expect(formatted).toContain('❌ FAIL API Token Valid');
            expect(formatted).toContain('Invalid token');
            expect(formatted).toContain('Recommendations:');
            expect(formatted).toContain('Your API token is invalid');
        });
    });
});
