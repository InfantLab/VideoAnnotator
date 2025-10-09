// Test token authentication specifically
const API_BASE = 'http://localhost:18011';
const TOKEN = 'dev-token';

async function testTokenAuth() {
    console.log('🔐 Testing token authentication...');

    const headers = {
        'Authorization': `Bearer ${TOKEN}`,
        'Content-Type': 'application/json'
    };

    try {
        console.log('📍 Testing /api/v1/debug/token-info endpoint...');
        const response = await fetch(`${API_BASE}/api/v1/debug/token-info`, { headers });

        if (response.ok) {
            const data = await response.json();
            console.log('✅ Token valid:', data);
        } else {
            console.log(`❌ Token invalid: ${response.status} ${response.statusText}`);

            if (response.status === 401) {
                console.log('🔍 Checking what token server expects...');
                // Try without auth to see error message
                const noAuthResponse = await fetch(`${API_BASE}/api/v1/debug/token-info`);
                const errorText = await noAuthResponse.text();
                console.log('Server response without auth:', errorText);
            }
        }
    } catch (error) {
        console.log(`💥 Request failed: ${error.message}`);
    }

    // Also test other working endpoints with same token
    try {
        console.log('📍 Testing /api/v1/pipelines/ with same token...');
        const pipelineResponse = await fetch(`${API_BASE}/api/v1/pipelines/`, { headers });
        console.log(`Pipelines endpoint: ${pipelineResponse.status} ${pipelineResponse.statusText}`);
    } catch (error) {
        console.log(`💥 Pipelines test failed: ${error.message}`);
    }
}

testTokenAuth();