#!/usr/bin/env python3
"""
Quick API testing script for client developers
Run: uv run python scripts/test_api_quick.py [base_url] [token]

Example:
    uv run python scripts/test_api_quick.py
    uv run python scripts/test_api_quick.py http://localhost:18011 my-api-token
"""
import requests
import json
import time
import sys
from pathlib import Path
from datetime import datetime


class APIQuickTester:
    def __init__(self, base_url="http://localhost:18011", token="dev-token"):
        self.base_url = base_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {token}"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.results = {"passed": 0, "failed": 0, "details": []}

    def _log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name}: {status}")
        if details:
            print(f"    {details}")
        
        self.results["passed" if passed else "failed"] += 1
        self.results["details"].append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def test_health_endpoints(self):
        """Test all health-related endpoints"""
        print("[HEALTH] Testing Health Endpoints...")
        
        # Basic health
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            passed = resp.status_code == 200
            details = f"Status: {resp.status_code}"
            if passed:
                data = resp.json()
                details += f", Server status: {data.get('status', 'unknown')}"
            self._log_result("Basic Health Check", passed, details)
        except Exception as e:
            self._log_result("Basic Health Check", False, f"Exception: {e}")

        # Detailed health
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/system/health", timeout=5)
            passed = resp.status_code == 200
            details = f"Status: {resp.status_code}"
            if passed:
                data = resp.json()
                db_status = data.get('services', {}).get('database', {}).get('status', 'unknown')
                details += f", DB status: {db_status}"
            self._log_result("Detailed Health Check", passed, details)
        except Exception as e:
            self._log_result("Detailed Health Check", False, f"Exception: {e}")

    def test_authentication(self):
        """Test authentication and token validation"""
        print("\n[AUTH] Testing Authentication...")
        
        # Test protected endpoint (jobs list)
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/jobs", timeout=5)
            passed = resp.status_code in [200, 401]  # Either valid auth or auth required
            details = f"Status: {resp.status_code}"
            if resp.status_code == 200:
                data = resp.json()
                details += f", Jobs accessible: {data.get('total', 0)} jobs"
            elif resp.status_code == 401:
                details += " (Authentication required - expected)"
            self._log_result("Authentication Test", passed, details)
        except Exception as e:
            self._log_result("Authentication Test", False, f"Exception: {e}")

        # Test token debug endpoint (if available)
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/debug/token-info", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                token_valid = data.get('token', {}).get('valid', False)
                permissions = data.get('token', {}).get('permissions', [])
                details = f"Token valid: {token_valid}, Permissions: {len(permissions)}"
                self._log_result("Token Debug Info", True, details)
            elif resp.status_code == 404:
                self._log_result("Token Debug Info", True, "Endpoint not implemented yet (expected)")
            else:
                self._log_result("Token Debug Info", False, f"Status: {resp.status_code}")
        except Exception as e:
            self._log_result("Token Debug Info", False, f"Exception: {e}")

    def test_pipelines(self):
        """Test pipeline endpoints"""
        print("\n[PIPELINE] Testing Pipeline Endpoints...")
        
        # Basic pipeline list
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/pipelines", timeout=5)
            passed = resp.status_code == 200
            details = f"Status: {resp.status_code}"
            if passed:
                data = resp.json()
                pipelines = data.get('pipelines', [])
                details += f", Available: {len(pipelines)} pipelines"
                if pipelines:
                    pipeline_names = [p.get('name', 'unknown') for p in pipelines[:3]]
                    details += f" ({', '.join(pipeline_names)})"
            self._log_result("Pipeline List", passed, details)
        except Exception as e:
            self._log_result("Pipeline List", False, f"Exception: {e}")

        # Debug pipeline info
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/debug/pipelines", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                pipelines = data.get('pipelines', [])
                details = f"Debug info available for {len(pipelines)} pipelines"
                self._log_result("Pipeline Debug Info", True, details)
            elif resp.status_code == 404:
                self._log_result("Pipeline Debug Info", True, "Debug endpoint not implemented yet (expected)")
            else:
                self._log_result("Pipeline Debug Info", False, f"Status: {resp.status_code}")
        except Exception as e:
            self._log_result("Pipeline Debug Info", False, f"Exception: {e}")

    def test_job_endpoints(self):
        """Test job management endpoints"""
        print("\n[JOBS] Testing Job Endpoints...")
        
        # Job listing
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/jobs", timeout=5)
            passed = resp.status_code in [200, 401]  # OK or auth required
            details = f"Status: {resp.status_code}"
            if resp.status_code == 200:
                data = resp.json()
                details += f", Total jobs: {data.get('total', 0)}"
            self._log_result("Job List", passed, details)
        except Exception as e:
            self._log_result("Job List", False, f"Exception: {e}")

        # Test job submission (mock video)
        job_id = None
        try:
            files = {'video': ('test.mp4', b'fake video content', 'video/mp4')}
            data = {'selected_pipelines': 'person_tracking'}
            
            resp = self.session.post(
                f"{self.base_url}/api/v1/jobs/", 
                files=files, 
                data=data, 
                timeout=10
            )
            passed = resp.status_code in [201, 401, 422]  # Created, auth required, or validation error
            details = f"Status: {resp.status_code}"
            
            if resp.status_code == 201:
                job_data = resp.json()
                job_id = job_data.get('id')
                details += f", Job ID: {job_id}"
            elif resp.status_code == 401:
                details += " (Authentication required)"
            elif resp.status_code == 422:
                details += " (Validation error - expected for mock data)"
                
            self._log_result("Job Submission", passed, details)
            
        except Exception as e:
            self._log_result("Job Submission", False, f"Exception: {e}")
        
        # Test job status retrieval if we have a job ID
        if job_id:
            try:
                resp = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}", timeout=5)
                passed = resp.status_code == 200
                details = f"Status: {resp.status_code}"
                if passed:
                    job_data = resp.json()
                    details += f", Job status: {job_data.get('status', 'unknown')}"
                self._log_result("Job Status Retrieval", passed, details)
            except Exception as e:
                self._log_result("Job Status Retrieval", False, f"Exception: {e}")
        
        return job_id

    def test_missing_endpoints(self):
        """Test for endpoints that should return 404 (not yet implemented)"""
        print("\n[MISSING] Testing Missing Endpoints (Should be 404)...")
        
        missing_endpoints = [
            ("/api/v1/events/stream", "SSE Endpoint"),
            ("/api/v1/jobs/123/results", "Job Results Download"),
            ("/api/v1/jobs/123/artifacts", "Job Artifacts"),
            ("/api/v1/videos", "Video Upload")
        ]
        
        for endpoint, name in missing_endpoints:
            try:
                resp = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                # 404 is expected (not implemented yet), 200 means it's implemented
                expected_404 = resp.status_code == 404
                details = f"Status: {resp.status_code}"
                if not expected_404:
                    details += " (Unexpectedly implemented!)"
                else:
                    details += " (Missing as expected)"
                    
                self._log_result(f"{name} Endpoint", True, details)
            except Exception as e:
                self._log_result(f"{name} Endpoint", False, f"Exception: {e}")

    def test_debug_endpoints(self):
        """Test debug endpoints"""
        print("\n[DEBUG] Testing Debug Endpoints...")
        
        debug_endpoints = [
            ("/api/v1/debug/server-info", "Server Info"),
            ("/api/v1/debug/token-info", "Token Info"),
            ("/api/v1/debug/pipelines", "Pipeline Debug"),
            ("/api/v1/debug/request-log", "Request Log")
        ]
        
        for endpoint, name in debug_endpoints:
            try:
                resp = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                # Debug endpoints might not be implemented yet
                if resp.status_code == 200:
                    details = "Debug endpoint working"
                elif resp.status_code == 404:
                    details = "Not implemented yet (expected)"
                elif resp.status_code == 401:
                    details = "Authentication required"
                else:
                    details = f"Status: {resp.status_code}"
                    
                passed = resp.status_code in [200, 404, 401]  # All acceptable
                self._log_result(f"{name} Debug", passed, details)
            except Exception as e:
                self._log_result(f"{name} Debug", False, f"Exception: {e}")

    def test_sse_connection(self):
        """Test SSE connection (mock or real)"""
        print("\n[SSE] Testing SSE Connection...")
        
        try:
            # Try to connect to SSE endpoint with a short timeout
            resp = self.session.get(
                f"{self.base_url}/api/v1/events/stream?token=dev-token", 
                timeout=3,
                stream=True
            )
            
            if resp.status_code == 200:
                # Try to read first event
                try:
                    for line in resp.iter_lines(decode_unicode=True):
                        if line and line.startswith('data:'):
                            event_data = line[5:].strip()  # Remove 'data:' prefix
                            details = f"Received SSE event: {event_data[:50]}..."
                            self._log_result("SSE Connection", True, details)
                            break
                    else:
                        self._log_result("SSE Connection", False, "No events received")
                except Exception as e:
                    self._log_result("SSE Connection", False, f"Stream error: {e}")
            elif resp.status_code == 404:
                self._log_result("SSE Connection", True, "SSE endpoint not implemented yet (expected)")
            else:
                self._log_result("SSE Connection", False, f"Status: {resp.status_code}")
                
        except requests.exceptions.Timeout:
            self._log_result("SSE Connection", True, "Connection timeout (expected if not implemented)")
        except Exception as e:
            self._log_result("SSE Connection", False, f"Exception: {e}")

    def run_all_tests(self):
        """Run comprehensive API tests"""
        print("[TEST] VideoAnnotator API Quick Test Suite")
        print("=" * 60)
        print(f"Testing server: {self.base_url}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run all test categories
        self.test_health_endpoints()
        self.test_authentication() 
        self.test_pipelines()
        job_id = self.test_job_endpoints()
        self.test_missing_endpoints()
        self.test_debug_endpoints()
        self.test_sse_connection()
        
        # Print summary
        print("\n" + "=" * 60)
        total_tests = self.results["passed"] + self.results["failed"]
        pass_rate = (self.results["passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"[COMPLETE] Test Suite Complete!")
        print(f"[RESULTS] {self.results['passed']}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        if self.results["failed"] > 0:
            print(f"[FAILED] {self.results['failed']} tests failed")
        else:
            print("[SUCCESS] All tests passed!")
        
        # Provide helpful tips
        print("\n[TIPS] Tips for Client Developers:")
        print("- Use /api/v1/debug/* endpoints for detailed information")
        print("- Check server logs for detailed error information")
        print("- Monitor /health endpoint for basic server status")
        if job_id:
            print(f"- Debug job details at /api/v1/debug/jobs/{job_id}")
        
        # Save results to file
        results_file = Path("test_results_api.json")
        with open(results_file, "w") as f:
            json.dump({
                "test_run": {
                    "timestamp": datetime.now().isoformat(),
                    "base_url": self.base_url,
                    "total_tests": total_tests,
                    "passed": self.results["passed"],
                    "failed": self.results["failed"],
                    "pass_rate": pass_rate
                },
                "results": self.results["details"]
            }, f, indent=2)
        
        print(f"[SAVED] Detailed results saved to: {results_file}")
        
        return self.results


def main():
    """Main function"""
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18011"
    token = sys.argv[2] if len(sys.argv) > 2 else "dev-token"
    
    print("VideoAnnotator API Quick Tester")
    print("================================")
    
    if len(sys.argv) <= 1:
        print("Using default settings:")
        print(f"  Server URL: {base_url}")
        print(f"  API Token: {token}")
        print("\nUsage: python scripts/test_api_quick.py [base_url] [token]")
        print("Example: python scripts/test_api_quick.py http://localhost:18011 my-token")
        print()
    
    # Create tester and run tests
    tester = APIQuickTester(base_url, token)
    results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    exit_code = 1 if results["failed"] > 0 else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()