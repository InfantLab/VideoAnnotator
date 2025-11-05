/**
 * Connection Diagnostics Utility
 * Helps quickly identify the root cause of connection failures
 */

export interface DiagnosticResult {
    test: string;
    passed: boolean;
    duration?: number;
    error?: string;
    details?: any;
}

export interface DiagnosticReport {
    timestamp: string;
    summary: 'all_passed' | 'partial_failure' | 'complete_failure';
    results: DiagnosticResult[];
    recommendations: string[];
}

/**
 * Run comprehensive connection diagnostics
 */
export async function runConnectionDiagnostics(apiUrl: string, token?: string): Promise<DiagnosticReport> {
    const results: DiagnosticResult[] = [];
    const recommendations: string[] = [];

    // Test 1: DNS Resolution
    try {
        const url = new URL(apiUrl);
        const hostname = url.hostname;

        if (hostname === 'localhost') {
            // First test localhost
            let localhostWorks = false;
            try {
                const response = await fetch(`${apiUrl}/api/v1/system/health`, {
                    method: 'GET',
                    signal: AbortSignal.timeout(3000)
                });
                localhostWorks = response.ok || response.status === 401 || response.status === 200;
            } catch {
                localhostWorks = false;
            }

            if (!localhostWorks) {
                // localhost failed, try 127.0.0.1
                const testUrl = apiUrl.replace('localhost', '127.0.0.1');
                const start = performance.now();

                try {
                    const response = await fetch(`${testUrl}/api/v1/system/health`, {
                        method: 'GET',
                        signal: AbortSignal.timeout(3000)
                    });
                    const duration = performance.now() - start;

                    if (response.ok || response.status === 401 || response.status === 200) {
                        results.push({
                            test: 'Server Address',
                            passed: false,
                            duration,
                            error: '"localhost" doesn\'t work, but direct IP address does',
                            details: { hostname }
                        });
                        recommendations.push('💡 Change server URL to: http://127.0.0.1:' + url.port);
                        recommendations.push('💡 Or fix your hosts file (see troubleshooting docs)');
                    } else {
                        // Both failed
                        results.push({
                            test: 'Server Address',
                            passed: false,
                            error: 'Cannot reach server at localhost or 127.0.0.1',
                            details: { hostname }
                        });
                    }
                } catch {
                    // Both failed
                    results.push({
                        test: 'Server Address',
                        passed: false,
                        error: 'Cannot reach server at localhost or 127.0.0.1',
                        details: { hostname }
                    });
                }
            } else {
                // localhost works
                results.push({
                    test: 'Server Address',
                    passed: true,
                    details: { hostname, note: 'localhost resolves correctly' }
                });
            }
        } else {
            results.push({
                test: 'Server Address',
                passed: true,
                details: { hostname }
            });
        }
    } catch (error) {
        results.push({
            test: 'Server Address',
            passed: false,
            error: error instanceof Error ? error.message : String(error)
        });
    }

    // Test 2: Can Reach Server
    try {
        const start = performance.now();
        const response = await fetch(`${apiUrl}/api/v1/system/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(5000)
        });
        const duration = performance.now() - start;

        results.push({
            test: 'Server Reachable',
            passed: response.ok || response.status === 401, // 401 means server is reachable
            duration,
            details: { status: response.status, statusText: response.statusText }
        });

        if (duration > 3000) {
            recommendations.push(`⏱️ Server is slow (${Math.round(duration / 1000)}s response time)`);
        }
    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        results.push({
            test: 'Server Reachable',
            passed: false,
            error: errorMsg
        });

        if (errorMsg.includes('timeout') || errorMsg.includes('timed out')) {
            recommendations.push('⏱️ Server not responding (timeout)');
            recommendations.push('💡 Check if VideoAnnotator server is running');
        } else if (errorMsg.includes('fetch')) {
            recommendations.push('🚫 Cannot connect to server');
            recommendations.push('💡 Check if firewall or antivirus is blocking connections');
        }
    }

    // Test 3: Cross-Origin Access (CORS)
    // Note: We test this with a real GET request since OPTIONS may return 405
    try {
        const start = performance.now();
        const response = await fetch(`${apiUrl}/api/v1/system/health`, {
            method: 'GET',
            headers: {
                'Origin': window.location.origin
            },
            signal: AbortSignal.timeout(5000)
        });
        const duration = performance.now() - start;

        const allowOrigin = response.headers.get('Access-Control-Allow-Origin');

        // If we got a response and can read headers, CORS is working
        const corsWorking = response.ok || response.status === 401;

        results.push({
            test: 'Cross-Origin Access',
            passed: corsWorking,
            duration,
            details: {
                status: response.status,
                allowOrigin: allowOrigin || '(not visible - but request succeeded)'
            }
        });

        if (!corsWorking) {
            recommendations.push('🚫 Server is blocking browser requests');
            recommendations.push('💡 If running VideoAnnotator locally: restart with "uv run videoannotator"');
        }
    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);

        // CORS errors often show as "Failed to fetch" or network errors
        if (errorMsg.includes('fetch') || errorMsg.includes('network') || errorMsg.includes('CORS')) {
            results.push({
                test: 'Cross-Origin Access',
                passed: false,
                error: 'Browser blocked the request'
            });
            recommendations.push('🚫 Server is blocking browser requests');
            recommendations.push('💡 If running VideoAnnotator locally: restart with "uv run videoannotator"');
        } else {
            results.push({
                test: 'Cross-Origin Access',
                passed: false,
                error: errorMsg
            });
        }
    }

    // Test 4: Token Authentication (if token provided)
    if (token) {
        try {
            const start = performance.now();
            const response = await fetch(`${apiUrl}/api/v1/jobs/?per_page=1`, {
                headers: { 'Authorization': `Bearer ${token}` },
                signal: AbortSignal.timeout(5000)
            });
            const duration = performance.now() - start;

            results.push({
                test: 'API Token Valid',
                passed: response.ok,
                duration,
                details: { status: response.status }
            });

            if (response.status === 401) {
                recommendations.push('🔑 Your API token is invalid or expired');
                recommendations.push('💡 Get a new token from the server administrator');
            } else if (response.status === 404) {
                recommendations.push('⚠️ Server returned 404 (may indicate auth issue on v1.3.0 servers)');
            }
        } catch (error) {
            results.push({
                test: 'API Token Valid',
                passed: false,
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }

    // Test 5: Server Version Detection
    try {
        const start = performance.now();
        const response = await fetch(`${apiUrl}/api/v1/system/health`, {
            signal: AbortSignal.timeout(5000)
        });
        const duration = performance.now() - start;

        if (response.ok) {
            const data = await response.json();
            results.push({
                test: 'Server Info',
                passed: true,
                duration,
                details: {
                    version: data.videoannotator_version || data.version,
                    api_version: data.api_version
                }
            });
        }
    } catch (error) {
        results.push({
            test: 'Server Info',
            passed: false,
            error: error instanceof Error ? error.message : String(error)
        });
    }

    // Determine overall summary
    const passedCount = results.filter(r => r.passed).length;
    const totalCount = results.length;

    let summary: DiagnosticReport['summary'];
    if (passedCount === totalCount) {
        summary = 'all_passed';
    } else if (passedCount > 0) {
        summary = 'partial_failure';
    } else {
        summary = 'complete_failure';
    }

    // Add general recommendations if complete failure
    if (summary === 'complete_failure') {
        recommendations.push('Server appears to be completely unreachable');
        recommendations.push('1. Verify server is running (PowerShell: Invoke-WebRequest http://localhost:18011/api/v1/system/health)');
        recommendations.push('2. Check if another application is using port 18011');
        recommendations.push('3. Try restarting the VideoAnnotator server');
    }

    return {
        timestamp: new Date().toISOString(),
        summary,
        results,
        recommendations: [...new Set(recommendations)] // Remove duplicates
    };
}

/**
 * Format diagnostic report as readable text
 */
export function formatDiagnosticReport(report: DiagnosticReport): string {
    let output = `Connection Diagnostics Report\n`;
    output += `Timestamp: ${new Date(report.timestamp).toLocaleString()}\n`;
    output += `Summary: ${report.summary.replace('_', ' ').toUpperCase()}\n\n`;

    output += `Test Results:\n`;
    output += `${'='.repeat(60)}\n`;

    for (const result of report.results) {
        const status = result.passed ? '✅ PASS' : '❌ FAIL';
        const duration = result.duration ? ` (${Math.round(result.duration)}ms)` : '';
        output += `${status} ${result.test}${duration}\n`;

        if (result.error) {
            output += `  Error: ${result.error}\n`;
        }

        if (result.details) {
            output += `  Details: ${JSON.stringify(result.details)}\n`;
        }

        output += '\n';
    }

    if (report.recommendations.length > 0) {
        output += `\nRecommendations:\n`;
        output += `${'='.repeat(60)}\n`;
        report.recommendations.forEach((rec, i) => {
            output += `${i + 1}. ${rec}\n`;
        });
    }

    return output;
}
