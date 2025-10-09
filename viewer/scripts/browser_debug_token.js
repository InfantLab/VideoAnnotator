// Browser Debug Console Script for Video Annotation Viewer
console.log("=== Video Annotation Viewer Debug ===");

// Check current localStorage settings
console.log("1. Current localStorage settings:");
const apiUrl = localStorage.getItem('videoannotator_api_url');
const apiToken = localStorage.getItem('videoannotator_api_token');
console.log(`API URL: ${apiUrl || 'NOT SET'}`);
console.log(`API Token: ${apiToken || 'NOT SET'}`);

// Check environment variables
console.log("\n2. Environment variables (if available):");
console.log(`VITE_API_BASE_URL: ${import.meta?.env?.VITE_API_BASE_URL || 'NOT SET'}`);
console.log(`VITE_API_TOKEN: ${import.meta?.env?.VITE_API_TOKEN || 'NOT SET'}`);

// Test API client configuration
console.log("\n3. Attempting to check API client config...");
try {
    // This might fail if apiClient is not available in global scope
    if (window.apiClient) {
        console.log("API Client found in global scope");
        console.log(`Base URL: ${window.apiClient.baseURL}`);
        console.log(`Token set: ${!!window.apiClient.token}`);
    } else {
        console.log("API Client not in global scope");
    }
} catch (e) {
    console.log("Error accessing API client:", e.message);
}

// Set correct values if needed
console.log("\n4. To fix localStorage, run these commands:");
console.log(`localStorage.setItem('videoannotator_api_url', 'http://localhost:18011');`);
console.log(`localStorage.setItem('videoannotator_api_token', 'dev-token');`);
console.log("Then refresh the page.");

// Test a simple fetch to see what error we get
console.log("\n5. Testing direct fetch...");
fetch('http://localhost:18011/api/v1/debug/token-info', {
    headers: {
        'Authorization': `Bearer ${apiToken || 'dev-token'}`
    }
})
    .then(response => {
        console.log(`Direct fetch status: ${response.status}`);
        if (response.status === 401) {
            console.log("❌ 401 Unauthorized - Token issue");
        } else if (response.status === 200) {
            console.log("✅ Direct fetch works - issue might be in client code");
        }
        return response.text();
    })
    .then(text => console.log(`Response: ${text.substring(0, 200)}...`))
    .catch(error => console.log(`Fetch error: ${error.message}`));