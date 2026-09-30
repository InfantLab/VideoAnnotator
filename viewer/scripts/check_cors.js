
async function checkCors(url) {
    console.log(`Checking CORS headers for ${url}...`);
    try {
        const response = await fetch(url, { method: 'OPTIONS' });
        console.log(`Status: ${response.status}`);
        console.log('Headers:');
        response.headers.forEach((val, key) => {
            if (key.includes('access-control')) {
                console.log(`  ${key}: ${val}`);
            }
        });
        
        // Also check GET
        const getResponse = await fetch(url);
        console.log('\nGET Response Headers:');
        getResponse.headers.forEach((val, key) => {
            if (key.includes('access-control')) {
                console.log(`  ${key}: ${val}`);
            }
        });
        
    } catch (error) {
        console.log(`Failed: ${error.message}`);
    }
}

async function run() {
    await checkCors('http://localhost:18011/api/v1/system/health');
}

run();
