/**
 * VideoAnnotator Browser Debug Console
 * 
 * Copy and paste this script into your browser console to enable
 * debugging tools for the VideoAnnotator API client.
 * 
 * Usage:
 * 1. Open browser developer tools (F12)
 * 2. Go to Console tab
 * 3. Paste this entire script and press Enter
 * 4. Use window.VideoAnnotatorDebug.* functions for debugging
 */

(function() {
    'use strict';
    
    // Configuration
    const API_BASE = window.location.origin; // Use current origin
    const DEFAULT_TOKEN = localStorage.getItem('api_token') || 'dev-token';
    
    console.log('🔧 VideoAnnotator Debug Console Loaded');
    console.log(`📡 API Base URL: ${API_BASE}`);
    
    // Create debug namespace
    window.VideoAnnotatorDebug = {
        // Configuration
        apiBase: API_BASE,
        defaultToken: DEFAULT_TOKEN,
        logRequests: true,
        
        // Quick API health check
        async checkHealth() {
            console.log('🏥 Checking API health...');
            try {
                const response = await fetch(`${this.apiBase}/health`);
                const data = await response.json();
                
                console.log('✅ Basic Health:', response.status === 200 ? 'OK' : 'FAIL');
                console.table({
                    'Status': data.status,
                    'API Version': data.api_version,
                    'Server': data.videoannotator_version
                });
                
                // Detailed health check
                const detailedResponse = await fetch(`${this.apiBase}/api/v1/system/health`);
                if (detailedResponse.ok) {
                    const detailedData = await detailedResponse.json();
                    console.log('✅ Detailed Health: OK');
                    console.table(detailedData.services);
                } else {
                    console.warn('⚠️ Detailed health check failed:', detailedResponse.status);
                }
                
                return data;
            } catch (error) {
                console.error('❌ Health check failed:', error);
                return null;
            }
        },

        // Test authentication 
        async checkAuth(token = null) {
            const authToken = token || this.defaultToken;
            console.log(`🔐 Testing authentication with token: ${authToken.substring(0, 10)}...`);
            
            try {
                const response = await fetch(`${this.apiBase}/api/v1/debug/token-info`, {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('✅ Authentication: Valid');
                    console.table({
                        'User ID': data.token?.user_id,
                        'Valid': data.token?.valid,
                        'Permissions': data.token?.permissions?.join(', '),
                        'Rate Limit': `${data.token?.rate_limit?.remaining_this_minute || '?'}/min remaining`
                    });
                    return data;
                } else if (response.status === 404) {
                    console.warn('⚠️ Debug endpoint not available yet');
                    return null;
                } else {
                    console.error('❌ Authentication failed:', response.status, await response.text());
                    return null;
                }
            } catch (error) {
                console.error('❌ Auth check failed:', error);
                return null;
            }
        },

        // Get server information
        async getServerInfo() {
            console.log('🖥️ Getting server information...');
            try {
                const response = await fetch(`${this.apiBase}/api/v1/debug/server-info`);
                if (response.ok) {
                    const data = await response.json();
                    console.log('✅ Server Info Retrieved');
                    
                    // Display key info in tables
                    console.group('📊 Server Status');
                    console.table(data.server);
                    console.table(data.system);
                    console.table(data.database);
                    console.groupEnd();
                    
                    return data;
                } else if (response.status === 404) {
                    console.warn('⚠️ Server info endpoint not available yet');
                    return null;
                } else {
                    console.error('❌ Failed to get server info:', response.status);
                    return null;
                }
            } catch (error) {
                console.error('❌ Server info request failed:', error);
                return null;
            }
        },

        // Test pipeline information
        async checkPipelines() {
            console.log('🔧 Checking pipeline information...');
            try {
                // Basic pipelines
                const basicResponse = await fetch(`${this.apiBase}/api/v1/pipelines`);
                if (basicResponse.ok) {
                    const basicData = await basicResponse.json();
                    console.log('✅ Basic Pipeline List: OK');
                    console.table(basicData.pipelines?.map(p => ({
                        Name: p.name,
                        Description: p.description
                    })) || []);
                }
                
                // Debug pipelines
                const debugResponse = await fetch(`${this.apiBase}/api/v1/debug/pipelines`);
                if (debugResponse.ok) {
                    const debugData = await debugResponse.json();
                    console.log('✅ Debug Pipeline Info: OK');
                    
                    debugData.pipelines?.forEach(pipeline => {
                        console.group(`🔧 ${pipeline.display_name || pipeline.name}`);
                        console.log('Status:', pipeline.status);
                        console.log('Components:', pipeline.components?.length || 0);
                        pipeline.components?.forEach(comp => {
                            console.log(`  - ${comp.name}: ${comp.enabled ? '✅' : '❌'} ${comp.model_loaded ? '(loaded)' : '(not loaded)'}`);
                        });
                        console.groupEnd();
                    });
                } else if (debugResponse.status === 404) {
                    console.warn('⚠️ Debug pipeline endpoint not available yet');
                }
                
                return { basic: basicData, debug: debugData };
            } catch (error) {
                console.error('❌ Pipeline check failed:', error);
                return null;
            }
        },

        // Monitor job progress
        async monitorJob(jobId, token = null) {
            const authToken = token || this.defaultToken;
            console.log(`📋 Monitoring job: ${jobId}`);
            
            const checkStatus = async () => {
                try {
                    const response = await fetch(`${this.apiBase}/api/v1/jobs/${jobId}`, {
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    
                    if (response.ok) {
                        const job = await response.json();
                        console.log(`📊 Job ${jobId} status: ${job.status}`, {
                            created: job.created_at,
                            started: job.started_at,
                            completed: job.completed_at
                        });
                        
                        if (job.status === 'completed' || job.status === 'failed') {
                            console.log('🏁 Job finished:', job);
                            return job;
                        }
                        
                        // Continue monitoring
                        setTimeout(checkStatus, 3000); // Check every 3 seconds
                    } else {
                        console.error('❌ Failed to get job status:', response.status);
                    }
                } catch (error) {
                    console.error('❌ Job monitoring error:', error);
                }
            };
            
            checkStatus();
        },

        // Test SSE connection
        async testSSE(jobId = null, token = null) {
            const authToken = token || this.defaultToken;
            const testJobId = jobId || 'test-job-123';
            
            console.log('📡 Testing SSE connection...');
            
            try {
                const url = `${this.apiBase}/api/v1/events/stream?token=${authToken}&job_id=${testJobId}`;
                const eventSource = new EventSource(url);
                
                let eventCount = 0;
                
                eventSource.onopen = () => {
                    console.log('✅ SSE Connection opened');
                };
                
                eventSource.onmessage = (event) => {
                    eventCount++;
                    try {
                        const data = JSON.parse(event.data);
                        console.log(`📡 SSE Event ${eventCount}:`, data);
                        
                        // Auto-close after 5 events for testing
                        if (eventCount >= 5) {
                            eventSource.close();
                            console.log('🔌 SSE connection closed after 5 events');
                        }
                    } catch (e) {
                        console.log(`📡 SSE Event ${eventCount} (raw):`, event.data);
                    }
                };
                
                eventSource.onerror = (error) => {
                    console.error('❌ SSE Connection error:', error);
                    if (eventSource.readyState === EventSource.CLOSED) {
                        console.log('🔌 SSE connection closed due to error');
                    }
                };
                
                // Auto-close after 15 seconds
                setTimeout(() => {
                    if (eventSource.readyState !== EventSource.CLOSED) {
                        eventSource.close();
                        console.log('🔌 SSE connection closed after timeout');
                        if (eventCount === 0) {
                            console.warn('⚠️ No SSE events received - endpoint may not be implemented yet');
                        }
                    }
                }, 15000);
                
                return eventSource;
            } catch (error) {
                console.error('❌ SSE test failed:', error);
                return null;
            }
        },

        // Submit test job
        async submitTestJob(token = null) {
            const authToken = token || this.defaultToken;
            console.log('📤 Submitting test job...');
            
            try {
                const formData = new FormData();
                const mockVideo = new Blob(['fake video content for testing'], { type: 'video/mp4' });
                formData.append('video', mockVideo, 'test-video.mp4');
                formData.append('selected_pipelines', 'person_tracking,scene_detection');
                
                const response = await fetch(`${this.apiBase}/api/v1/jobs/`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${authToken}` },
                    body: formData
                });
                
                if (response.ok) {
                    const job = await response.json();
                    console.log('✅ Test job submitted:', job.id);
                    console.table({
                        'Job ID': job.id,
                        'Status': job.status,
                        'Video Path': job.video_path,
                        'Pipelines': job.selected_pipelines?.join(', ')
                    });
                    return job;
                } else {
                    const error = await response.text();
                    console.error('❌ Job submission failed:', response.status, error);
                    return null;
                }
            } catch (error) {
                console.error('❌ Test job submission failed:', error);
                return null;
            }
        },

        // Run comprehensive tests
        async runAllTests(token = null) {
            console.log('🧪 Running comprehensive API tests...');
            console.log('=' * 50);
            
            const results = {
                health: await this.checkHealth(),
                auth: await this.checkAuth(token),
                serverInfo: await this.getServerInfo(), 
                pipelines: await this.checkPipelines(),
                testJob: await this.submitTestJob(token)
            };
            
            console.log('=' * 50);
            console.log('📊 Test Results Summary:');
            Object.entries(results).forEach(([test, result]) => {
                console.log(`- ${test}: ${result ? '✅' : '❌'}`);
            });
            
            if (results.testJob) {
                console.log('\n💡 You can monitor the test job with:');
                console.log(`VideoAnnotatorDebug.monitorJob('${results.testJob.id}')`);
            }
            
            return results;
        },

        // Enable request logging
        enableRequestLogging() {
            if (this._originalFetch) {
                console.log('⚠️ Request logging is already enabled');
                return;
            }
            
            console.log('🔍 Enabling API request logging...');
            this._originalFetch = window.fetch;
            
            window.fetch = (...args) => {
                const [url, options = {}] = args;
                if (this.logRequests && url.includes('/api/')) {
                    console.group('📤 API Request');
                    console.log('URL:', url);
                    console.log('Method:', options.method || 'GET');
                    if (options.headers) console.log('Headers:', options.headers);
                    console.groupEnd();
                }
                
                return this._originalFetch.apply(window, args)
                    .then(response => {
                        if (this.logRequests && url.includes('/api/')) {
                            console.group('📥 API Response');
                            console.log('URL:', url);
                            console.log('Status:', response.status, response.statusText);
                            console.groupEnd();
                        }
                        return response;
                    });
            };
        },

        // Disable request logging
        disableRequestLogging() {
            if (this._originalFetch) {
                console.log('🔕 Disabling API request logging...');
                window.fetch = this._originalFetch;
                delete this._originalFetch;
            }
        },

        // Show help
        help() {
            console.log('🔧 VideoAnnotator Debug Console Help');
            console.log('=====================================');
            console.log('Available commands:');
            console.log('');
            console.log('• checkHealth()           - Test API health endpoints');
            console.log('• checkAuth(token)        - Test authentication');
            console.log('• getServerInfo()         - Get server debug information');
            console.log('• checkPipelines()        - Check pipeline status');
            console.log('• monitorJob(jobId)       - Monitor job progress');
            console.log('• testSSE(jobId)          - Test Server-Sent Events');
            console.log('• submitTestJob()         - Submit a test job');
            console.log('• runAllTests()           - Run comprehensive test suite');
            console.log('• enableRequestLogging()  - Log all API requests');
            console.log('• disableRequestLogging() - Stop logging requests');
            console.log('• help()                  - Show this help message');
            console.log('');
            console.log('Examples:');
            console.log('  VideoAnnotatorDebug.runAllTests()');
            console.log('  VideoAnnotatorDebug.checkHealth()');
            console.log('  VideoAnnotatorDebug.monitorJob("job_123")');
            console.log('');
            console.log('Configuration:');
            console.log(`  API Base: ${this.apiBase}`);
            console.log(`  Default Token: ${this.defaultToken}`);
            console.log(`  Request Logging: ${this.logRequests}`);
        }
    };

    // Auto-enable request logging
    window.VideoAnnotatorDebug.enableRequestLogging();
    
    // Show help on load
    console.log('');
    console.log('💡 Type VideoAnnotatorDebug.help() for available commands');
    console.log('🚀 Quick start: VideoAnnotatorDebug.runAllTests()');
    console.log('');
    
})();