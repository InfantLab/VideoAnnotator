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

// Helper for timeout signal (safe for older browsers)
function timeoutSignal(ms: number): AbortSignal {
    // Use AbortController explicitly to match test_strict_connection.html behavior
    const controller = new AbortController();
    setTimeout(() => controller.abort(), ms);
    return controller.signal;
}

/**
 * Run comprehensive connection diagnostics
 */
export async function runConnectionDiagnostics(apiUrl: string, token?: string): Promise<DiagnosticReport> {
    const results: DiagnosticResult[] = [];
    const recommendations: string[] = [];

    console.log('🔍 DIAGNOSTICS STARTED');
    console.log('Target API URL:', apiUrl);

    // Normalize API URL
    const cleanApiUrl = apiUrl.replace(/\/$/, '');
    const isRelative = !cleanApiUrl.startsWith('http');

    // Helper to get hostname
    let targetHostname = 'unknown';
    let targetPort = '';
    try {
        if (!isRelative) {
            const url = new URL(cleanApiUrl);
            targetHostname = url.hostname;
            targetPort = url.port;
        } else {
            targetHostname = window.location.hostname;
            targetPort = window.location.port;
        }
        console.log('Parsed Hostname:', targetHostname, 'Port:', targetPort);
    } catch (e) {
        // Invalid URL
        results.push({
            test: 'URL Validation',
            passed: false,
            error: 'Invalid API URL format'
        });
        return {
            timestamp: new Date().toISOString(),
            summary: 'complete_failure',
            results,
            recommendations: ['Check the API URL format']
        };
    }

    // Test 1: Server Reachability
    let serverReachable = false;
    let serverResponse: Response | null = null;
    const testUrl = `${cleanApiUrl}/api/v1/system/health`;

    try {
        const start = performance.now();
        console.log(`Attempting fetch: ${testUrl}`);
        
        // Try the configured URL first
        const response = await fetch(testUrl, {
            method: 'GET',
            signal: timeoutSignal(10000) // Increased to 10s
        });
        const duration = performance.now() - start;
        console.log(`Fetch success (${duration}ms):`, response.status);
        serverResponse = response;

        serverReachable = response.ok || response.status === 401;

        results.push({
            test: 'Server Reachable',
            passed: serverReachable,
            duration,
            details: {
                status: response.status,
                statusText: response.statusText,
                url: testUrl,
                mode: isRelative ? 'Proxy (Relative URL)' : 'Direct (Absolute URL)'
            }
        });

        if (duration > 3000) {
            recommendations.push(`⏱️ Server is slow (${Math.round(duration / 1000)}s response time)`);
        }

    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);

        results.push({
            test: 'Server Reachable',
            passed: false,
            error: errorMsg,
            details: { url: `${cleanApiUrl}/api/v1/system/health` }
        });

        if (errorMsg.includes('timeout') || errorMsg.includes('timed out') || errorMsg.includes('abort')) {
            recommendations.push('⏱️ Server not responding (timeout)');
            recommendations.push('💡 Check if VideoAnnotator server is running');
        } else if (errorMsg.includes('fetch') || errorMsg.includes('Failed to fetch')) {
            // Could be CORS or Connection Refused
            recommendations.push('🚫 Cannot connect to server (Connection Refused or CORS)');
        }
    }

    // Test 2: Address Resolution (localhost vs 127.0.0.1)
    // Only relevant if we failed to reach the server and we are using localhost
    if (!serverReachable && targetHostname === 'localhost' && !isRelative) {
        const altUrl = cleanApiUrl.replace('localhost', '127.0.0.1');
        try {
            const res = await fetch(`${altUrl}/api/v1/system/health`, { signal: timeoutSignal(3000) });
            if (res.ok || res.status === 401) {
                results.push({
                    test: 'Address Resolution',
                    passed: false,
                    error: 'localhost failed, but 127.0.0.1 worked',
                    details: { note: 'IPv4/IPv6 resolution issue' }
                });
                recommendations.push(`💡 Use IP address instead of localhost: ${altUrl}`);
            } else {
                results.push({
                    test: 'Address Resolution',
                    passed: false,
                    error: 'Both localhost and 127.0.0.1 failed'
                });
            }
        } catch (e) {
            results.push({
                test: 'Address Resolution',
                passed: false,
                error: 'Both localhost and 127.0.0.1 failed'
            });
        }
    } else if (serverReachable) {
        results.push({
            test: 'Address Resolution',
            passed: true,
            details: { hostname: targetHostname }
        });
    }

    // Test 3: Proxy Check (if direct failed and we are in dev mode/localhost)
    if (!serverReachable && !isRelative && window.location.hostname === 'localhost') {
        try {
            // Try relative path which goes through Vite proxy
            // We assume the proxy is set up at root or /api
            // Let's try fetching relative to current origin
            const proxyUrl = '/api/v1/system/health';
            const res = await fetch(proxyUrl, { signal: timeoutSignal(3000) });
            
            if (res.ok || res.status === 401) {
                results.push({
                    test: 'Proxy Availability',
                    passed: true,
                    details: { note: 'Proxy works!' }
                });
                recommendations.push('💡 Connection blocked? Try using the built-in proxy by clearing the API URL field (leave it empty).');
            }
        } catch (e) {
            // Proxy also failed, no need to report
        }
    }

    // Test 4: Cross-Origin Access (CORS)
    // We can only definitively test this if the server is reachable but blocks us,
    // OR if we can reach it via a mode that doesn't use CORS (not possible in browser),
    // OR if we inspect the error carefully.
    if (serverReachable && !isRelative) {
        // If server is reachable, check headers
        const allowOrigin = serverResponse?.headers.get('Access-Control-Allow-Origin');
        results.push({
            test: 'Cross-Origin Access',
            passed: true, // If we got a response, CORS is technically working or not blocking this request
            details: {
                allowOrigin: allowOrigin || '(not visible)'
            }
        });
    } else if (!serverReachable && !isRelative) {
        // If fetch failed, it MIGHT be CORS.
        // We can't distinguish "Connection Refused" from "CORS Error" easily in JS.
        // But if "Address Resolution" (127.0.0.1) also failed, it's likely Connection Refused.
        results.push({
            test: 'Cross-Origin Access',
            passed: false,
            error: 'Cannot test (Server unreachable)'
        });
    }

    // Test 5: Token Authentication (if token provided)
    if (token && serverReachable) {
        try {
            const start = performance.now();
            // NOTE: Trailing slash is required by server for this endpoint
            const response = await fetch(`${cleanApiUrl}/api/v1/jobs/?per_page=1`, {
                headers: { 'Authorization': `Bearer ${token}` },
                signal: timeoutSignal(10000)
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
    } else if (token) {
         results.push({
            test: 'API Token Valid',
            passed: false,
            error: 'Skipped (Server unreachable)'
        });
    }

    // Test 6: Server Version Detection
    if (serverReachable) {
        try {
            const start = performance.now();
            // We already fetched health, reuse if possible or fetch again
            let data;
            if (serverResponse && !serverResponse.bodyUsed) {
                 data = await serverResponse.json();
            } else {
                 const res = await fetch(`${cleanApiUrl}/api/v1/system/health`, { signal: timeoutSignal(10000) });
                 data = await res.json();
            }
            const duration = performance.now() - start;

            results.push({
                test: 'Server Info',
                passed: true,
                duration,
                details: {
                    version: data.videoannotator_version || data.version,
                    api_version: data.api_version
                }
            });
        } catch (error) {
            results.push({
                test: 'Server Info',
                passed: false,
                error: error instanceof Error ? error.message : String(error)
            });
        }
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
        recommendations.push(`1. Verify server is running on port ${targetPort || '18011'}`);
        recommendations.push('2. Check if firewall is blocking connections');
        if (targetHostname === 'localhost') {
             recommendations.push('3. Try using 127.0.0.1 instead of localhost');
        }
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
