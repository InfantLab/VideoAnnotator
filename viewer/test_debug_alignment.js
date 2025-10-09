// Test to verify debugUtils now uses the same config as API client
console.log('🔧 Testing debug utils token alignment...');

// Check what debugUtils would use
if (typeof window !== 'undefined' && window.VideoAnnotatorDebug) {
  console.log('🎯 VideoAnnotatorDebug config:');
  console.log('  API Base:', window.VideoAnnotatorDebug.apiBase);
  console.log('  Token:', window.VideoAnnotatorDebug.defaultToken.substring(0, 10) + '...');
} else {
  console.log('❌ VideoAnnotatorDebug not available');
}

// Check localStorage directly
console.log('\n📦 Current localStorage:');
console.log('  videoannotator_api_token:', localStorage.getItem('videoannotator_api_token'));
console.log('  videoannotator_api_url:', localStorage.getItem('videoannotator_api_url'));

// Test authentication with current config
async function testCurrentConfig() {
  const token = localStorage.getItem('videoannotator_api_token') || 'dev-token';
  const url = localStorage.getItem('videoannotator_api_url') || 'http://localhost:18011';

  console.log(`\n🧪 Testing: ${url}/api/v1/debug/token-info with token: ${token.substring(0, 10)}...`);

  try {
    const response = await fetch(`${url}/api/v1/debug/token-info`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.ok) {
      const data = await response.json();
      console.log('✅ Token works!', data.token?.user_id);
    } else {
      console.log(`❌ Token failed: ${response.status}`);
    }
  } catch (error) {
    console.log('💥 Request failed:', error.message);
  }
}

testCurrentConfig();