import json

import requests


def investigate_server_truth():
    """
    Instead of assuming what endpoints should exist, let's ask the server
    what it actually provides and align our client with that reality.
    """
    
    base_url = "http://localhost:18011"
    headers = {"Authorization": "Bearer dev-token"}
    
    print("🔍 INVESTIGATING SERVER'S ACTUAL CAPABILITIES")
    print("=" * 60)
    
    # 1. What does the server say about itself?
    print("1. Server Self-Reported Information:")
    print("-" * 40)
    
    # Check what the server tells us in working endpoints
    working_info_endpoints = [
        ("/health", "Basic health check"),
        ("/api/v1/system/health", "System health"),
        ("/api/v1/debug/server-info", "Debug server info")
    ]
    
    server_capabilities = {}
    
    for endpoint, desc in working_info_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers)
            if response.status_code == 200:
                data = response.json()
                server_capabilities[endpoint] = data
                print(f"✅ {endpoint}: {desc}")
                
                # Extract key info
                if 'api_version' in data:
                    print(f"   📊 API Version: {data['api_version']}")
                if 'version' in data:
                    print(f"   📊 Server Version: {data['version']}")
                if 'videoannotator_version' in data:
                    print(f"   📊 VideoAnnotator Version: {data['videoannotator_version']}")
                if 'features' in data:
                    print(f"   🔧 Features: {data['features']}")
                if 'capabilities' in data:
                    print(f"   ⚙️  Capabilities: {data['capabilities']}")
                    
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
    
    print()
    
    # 2. What endpoints does the server actually expose?
    print("2. Server's Actual Available Endpoints:")
    print("-" * 40)
    
    # Check if server provides any endpoint discovery
    discovery_endpoints = [
        "/api/v1/debug/routes",
        "/api/v1/debug/endpoints", 
        "/api/v1/openapi.json",
        "/docs",
        "/openapi.json",
        "/.well-known/endpoints"
    ]
    
    available_endpoints = []
    
    for endpoint in discovery_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers)
            if response.status_code == 200:
                print(f"✅ {endpoint}: Endpoint discovery available")
                if endpoint.endswith('.json'):
                    try:
                        data = response.json()
                        if 'paths' in data:  # OpenAPI spec
                            paths = list(data['paths'].keys())
                            print(f"   📋 Available paths: {len(paths)} endpoints")
                            available_endpoints.extend(paths)
                        else:
                            print(f"   📋 Data keys: {list(data.keys())}")
                    except:
                        pass
            else:
                print(f"❌ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
    
    print()
    
    # 3. What does the server say about v1.2.x features?
    print("3. V1.2.x Feature Analysis:")
    print("-" * 40)
    
    # Look at debug server info for clues
    if '/api/v1/debug/server-info' in server_capabilities:
        debug_info = server_capabilities['/api/v1/debug/server-info']
        print("Server debug info analysis:")
        
        if 'server' in debug_info:
            server_info = debug_info['server']
            api_version = server_info.get('version', 'unknown')
            videoannotator_version = server_info.get('videoannotator_version', 'unknown')
            
            print(f"   📊 API Version: {api_version}")
            print(f"   📊 VideoAnnotator Version: {videoannotator_version}")
            
            # Determine what features should be available based on actual version
            if api_version.startswith('1.2'):
                print("   ✅ API v1.2.x detected - should have v1.2.x endpoints")
            elif api_version.startswith('1.1') or api_version.startswith('1.0'):
                print("   ℹ️  API v1.0/1.1 detected - v1.2.x endpoints not expected")
            else:
                print(f"   ❓ Unknown API version pattern: {api_version}")
    
    print()
    
    # 4. Client Alignment Recommendations
    print("4. CLIENT ALIGNMENT RECOMMENDATIONS:")
    print("-" * 40)
    
    # Based on what server actually provides
    if '/health' in server_capabilities:
        health_data = server_capabilities['/health']
        actual_api_version = health_data.get('api_version', 'unknown')
        
        print(f"Server reports API version: {actual_api_version}")
        
        if actual_api_version == '1.2.0':
            print("✅ RECOMMENDATION: Client should target API v1.2.0 features")
            print("   - Use /api/v1/debug/server-info for server info (works)")
            print("   - Don't expect /api/v1/system/server-info (doesn't exist)")
            print("   - Don't expect /api/v1/pipelines/catalog (doesn't exist)")
            print("   - Use /api/v1/pipelines for pipeline list (works)")
        elif actual_api_version.startswith('1.1') or actual_api_version.startswith('1.0'):
            print("✅ RECOMMENDATION: Client should target API v1.0/1.1 features")
            print("   - Use legacy endpoints only")
            print("   - Don't attempt v1.2.x endpoints")
        else:
            print(f"❓ UNKNOWN API VERSION: {actual_api_version}")
            print("   - Client should use only confirmed working endpoints")
    
    print()
    print("5. SUMMARY:")
    print("-" * 40)
    print("The client should:")
    print("- Query server capabilities at startup")
    print("- Adapt to server's actual API version") 
    print("- Not assume endpoints exist without confirmation")
    print("- Use server's self-reported capabilities as source of truth")
    
    # Save findings
    with open("server_truth_analysis.json", "w") as f:
        json.dump({
            "timestamp": "2025-09-26",
            "analysis": "Server truth investigation",
            "capabilities": server_capabilities,
            "available_endpoints": available_endpoints,
            "recommendations": "Align client with server's actual API version"
        }, f, indent=2)
    
    print(f"\n📁 Detailed analysis saved to: server_truth_analysis.json")

if __name__ == "__main__":
    investigate_server_truth()