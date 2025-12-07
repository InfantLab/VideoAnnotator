
async function testEndpoint() {
    const baseUrl = 'http://127.0.0.1:18011';
    const token = 'dev-token';
    
    console.log(`Testing ${baseUrl} with token: ${token}`);

    const endpoints = [
        '/api/v1/pipelines',
        '/api/v1/pipelines/',
    ];

    for (const endpoint of endpoints) {
        const url = `${baseUrl}${endpoint}`;
        console.log(`\nFetching ${url}...`);
        try {
            const start = performance.now();
            const res = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Accept': 'application/json'
                }
            });
            const duration = Math.round(performance.now() - start);
            console.log(`Status: ${res.status} ${res.statusText} (${duration}ms)`);
            const text = await res.text();
            console.log(`Body preview: ${text.substring(0, 200)}`);
        } catch (e) {
            console.error(`Error: ${e.message}`);
        }
    }
}

testEndpoint();
