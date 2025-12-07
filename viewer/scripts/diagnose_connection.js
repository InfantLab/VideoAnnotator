
async function testUrl(url) {
    console.log(`Testing ${url}...`);
    try {
        const start = performance.now();
        const response = await fetch(url, { signal: AbortSignal.timeout(5000) });
        const duration = performance.now() - start;
        
        console.log(`✅ Status: ${response.status} ${response.statusText}`);
        console.log(`   Time: ${Math.round(duration)}ms`);
        console.log(`   Headers:`);
        response.headers.forEach((val, key) => console.log(`     ${key}: ${val}`));
        
        try {
            const text = await response.text();
            console.log(`   Body preview: ${text.substring(0, 100)}...`);
        } catch (e) {
            console.log(`   Could not read body: ${e.message}`);
        }
        return true;
    } catch (error) {
        console.log(`❌ Failed: ${error.message}`);
        if (error.cause) console.log(`   Cause: ${error.cause}`);
        return false;
    }
}

async function run() {
    console.log("=== Connection Diagnostic Script ===");
    
    console.log("\n1. Testing localhost (IPv4/IPv6)...");
    await testUrl('http://localhost:18011/health');
    
    console.log("\n2. Testing 127.0.0.1 (IPv4)...");
    await testUrl('http://127.0.0.1:18011/health');

    console.log("\n3. Testing API endpoint (localhost)...");
    await testUrl('http://localhost:18011/api/v1/system/health');
}

run();
