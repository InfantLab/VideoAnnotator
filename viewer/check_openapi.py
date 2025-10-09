import json

import requests

# Get the actual OpenAPI spec from the server
response = requests.get("http://localhost:18011/openapi.json")

if response.status_code == 200:
    openapi_spec = response.json()
    
    print("📋 SERVER'S ACTUAL API ENDPOINTS (from OpenAPI spec):")
    print("=" * 60)
    
    paths = openapi_spec.get('paths', {})
    
    # Group endpoints by category
    system_endpoints = []
    pipeline_endpoints = []
    job_endpoints = []
    debug_endpoints = []
    other_endpoints = []
    
    for path in sorted(paths.keys()):
        if '/system/' in path:
            system_endpoints.append(path)
        elif '/pipeline' in path:
            pipeline_endpoints.append(path)
        elif '/job' in path:
            job_endpoints.append(path)
        elif '/debug/' in path:
            debug_endpoints.append(path)
        else:
            other_endpoints.append(path)
    
    print(f"📊 SYSTEM ENDPOINTS ({len(system_endpoints)}):")
    for endpoint in system_endpoints:
        print(f"   ✅ {endpoint}")
    
    print(f"\n🔄 PIPELINE ENDPOINTS ({len(pipeline_endpoints)}):")
    for endpoint in pipeline_endpoints:
        print(f"   ✅ {endpoint}")
    
    print(f"\n💼 JOB ENDPOINTS ({len(job_endpoints)}):")
    for endpoint in job_endpoints:
        print(f"   ✅ {endpoint}")
    
    print(f"\n🔧 DEBUG ENDPOINTS ({len(debug_endpoints)}):")
    for endpoint in debug_endpoints:
        print(f"   ✅ {endpoint}")
    
    print(f"\n📝 OTHER ENDPOINTS ({len(other_endpoints)}):")
    for endpoint in other_endpoints:
        print(f"   ✅ {endpoint}")
    
    print(f"\n📋 TOTAL ENDPOINTS: {len(paths)}")
    
    # Now check what the client is trying to access vs what exists
    print("\n" + "=" * 60)
    print("🔍 CLIENT vs SERVER ENDPOINT ANALYSIS:")
    print("=" * 60)
    
    client_expected = [
        '/api/v1/system/info',
        '/api/v1/system/server-info',
        '/api/v1/pipelines/catalog',
        '/api/v1/pipelines/schema'
    ]
    
    print("Client expects these endpoints (causing 404s):")
    for endpoint in client_expected:
        if endpoint in paths:
            print(f"   ✅ {endpoint} - EXISTS")
        else:
            print(f"   ❌ {endpoint} - MISSING (404 expected)")
    
    print("\nServer actually provides:")
    for endpoint in sorted(paths.keys()):
        if any(expected in endpoint for expected in ['/system/', '/pipeline']):
            print(f"   ✅ {endpoint}")
    
    # Save the actual API spec
    with open("server_actual_api_spec.json", "w") as f:
        json.dump(openapi_spec, f, indent=2)
    
    print(f"\n📁 Full OpenAPI spec saved to: server_actual_api_spec.json")
    
else:
    print(f"❌ Could not fetch OpenAPI spec: {response.status_code}")