// Browser console script to debug and fix token issues
// Paste this into the browser dev console (F12 -> Console)

console.log('🔍 Token Configuration Debug:');
console.log('='.repeat(50));

// Check localStorage
console.log('📦 localStorage values:');
console.log('  videoannotator_api_url:', localStorage.getItem('videoannotator_api_url'));
console.log('  videoannotator_api_token:', localStorage.getItem('videoannotator_api_token'));

// Check environment variables (if available in dev)
console.log('🌍 Environment variables:');
console.log('  VITE_API_BASE_URL:', import.meta?.env?.VITE_API_BASE_URL || 'not available');
console.log('  VITE_API_TOKEN:', import.meta?.env?.VITE_API_TOKEN || 'not available');

// Check what the API client is actually using
const getApiToken = () => {
    return localStorage.getItem('videoannotator_api_token') ||
        import.meta?.env?.VITE_API_TOKEN ||
        'dev-token';
};

console.log('🎯 API client will use token:', getApiToken());

// Fix: Clear localStorage and set correct token
console.log('🔧 Fixing token configuration...');
localStorage.setItem('videoannotator_api_token', 'dev-token');
localStorage.setItem('videoannotator_api_url', 'http://localhost:18011');

console.log('✅ Fixed! localStorage now contains:');
console.log('  videoannotator_api_url:', localStorage.getItem('videoannotator_api_url'));
console.log('  videoannotator_api_token:', localStorage.getItem('videoannotator_api_token'));

console.log('🔄 Please refresh the page for changes to take effect.');