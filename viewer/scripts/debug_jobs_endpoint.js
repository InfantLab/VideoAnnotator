
async function testEndpoint() {
    const baseUrl = 'http://127.0.0.1:18011';
    const token = 'dev-token';
    
    console.log(`Testing ${baseUrl} with token: ${token}`);

    const endpoints = [
        '/api/v1/system/health',
        '/api/v1/jobs',
        '/api/v1/jobs/',
        '/api/v1/jobs?per_page=1'
    ];

    for (const endpoint of endpoints) {
        const url = `${baseUrl}${endpoint}`;
        console.log(`\nFetching ${url}...`);
        try {
            const res = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Accept': 'application/json'
                }
            });
            console.log(`Status: ${res.status} ${res.statusText}`);
            const text = await res.text();
            console.log(`Body preview: ${text.substring(0, 200)}`);
        } catch (e) {
            console.error(`Error: ${e.message}`);
        }
    }
}

testEndpoint();
