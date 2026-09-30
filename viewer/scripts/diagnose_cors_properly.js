
async function testPreflight(url, origin) {
    console.log(`\nTesting Preflight (OPTIONS) for ${url}`);
    console.log(`Origin: ${origin}`);
    
    try {
        const response = await fetch(url, { 
            method: 'OPTIONS',
            headers: {
                'Origin': origin,
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'authorization,content-type'
            }
        });
        
        console.log(`Status: ${response.status} ${response.statusText}`);
        
        const allowOrigin = response.headers.get('access-control-allow-origin');
        const allowMethods = response.headers.get('access-control-allow-methods');
        
        console.log(`Access-Control-Allow-Origin: ${allowOrigin || '(missing)'}`);
        console.log(`Access-Control-Allow-Methods: ${allowMethods || '(missing)'}`);
        
        if (response.ok && allowOrigin) {
            console.log("✅ CORS Preflight Passed");
            return true;
        } else {
            console.log("❌ CORS Preflight Failed");
            return false;
        }
    } catch (error) {
        console.log(`❌ Network Error: ${error.message}`);
        return false;
    }
}

async function run() {
    // Test with the Vite default port origin
    await testPreflight('http://localhost:18011/api/v1/system/health', 'http://localhost:5173');
    
    // Test with a generic origin
    await testPreflight('http://localhost:18011/api/v1/system/health', 'http://example.com');
}

run();
