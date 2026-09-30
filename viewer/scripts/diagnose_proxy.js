
async function testUrl(url) {
    console.log(`Testing ${url}...`);
    try {
        const start = performance.now();
        const response = await fetch(url, { signal: AbortSignal.timeout(5000) });
        const duration = performance.now() - start;
        
        console.log(`✅ Status: ${response.status} ${response.statusText}`);
        console.log(`   Time: ${Math.round(duration)}ms`);
        return true;
    } catch (error) {
        console.log(`❌ Failed: ${error.message}`);
        if (error.cause) console.log(`   Cause: ${error.cause}`);
        return false;
    }
}

async function run() {
    console.log("=== Proxy Diagnostic Script ===");
    console.log("Assuming Vite is running on port 19011...");
    
    // Test the proxy endpoint
    await testUrl('http://localhost:19011/api/v1/system/health');
    
    // Test the proxy health endpoint
    await testUrl('http://localhost:19011/health');
}

run();
