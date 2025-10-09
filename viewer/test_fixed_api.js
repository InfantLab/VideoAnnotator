// Quick API test to verify fixes
const API_BASE = 'http://localhost:18011';
const TOKEN = 'dev-token';

async function testAPI() {
    console.log('🧪 Testing fixed API endpoints...');

    const headers = {
        'Authorization': `Bearer ${TOKEN}`,
        'Content-Type': 'application/json'
    };

    // Test working endpoints
    const tests = [
        { name: 'Health Check', url: `${API_BASE}/health` },
        { name: 'System Health', url: `${API_BASE}/api/v1/system/health` },
        { name: 'Debug Server Info', url: `${API_BASE}/api/v1/debug/server-info` },
        { name: 'Pipelines List', url: `${API_BASE}/api/v1/pipelines/` }
    ];

    for (const test of tests) {
        try {
            const response = await fetch(test.url, { headers });
            if (response.ok) {
                const data = await response.json();
                console.log(`✅ ${test.name}: ${response.status}`);
                if (test.name === 'Pipelines List') {
                    console.log(`   📊 Found ${data.pipelines ? data.pipelines.length : Array.isArray(data) ? data.length : 'unknown'} pipelines`);
                }
            } else {
                console.log(`❌ ${test.name}: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.log(`💥 ${test.name}: ${error.message}`);
        }
    }

    // Test endpoints that should 404
    console.log('\n🚫 Testing endpoints that should fail:');
    const shouldFail = [
        { name: 'System Info (missing)', url: `${API_BASE}/api/v1/system/info` },
        { name: 'Pipeline Catalog (missing)', url: `${API_BASE}/api/v1/pipelines/catalog` }
    ];

    for (const test of shouldFail) {
        try {
            const response = await fetch(test.url, { headers });
            if (response.status === 404) {
                console.log(`✅ ${test.name}: 404 (expected)`);
            } else {
                console.log(`⚠️ ${test.name}: ${response.status} (unexpected)`);
            }
        } catch (error) {
            console.log(`💥 ${test.name}: ${error.message}`);
        }
    }
}

testAPI();