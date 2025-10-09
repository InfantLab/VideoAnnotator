"""
Job Failure Diagnostics Script
Collects information about failed jobs for troubleshooting

Usage:
    python scripts/diagnose_failed_jobs.py

Requirements:
    - VideoAnnotator server running at localhost:18011
    - Valid API token (default: dev-token)
"""

import json
from datetime import datetime
from pathlib import Path

import requests

API_BASE = "http://localhost:18011"
TOKEN = "dev-token"
OUTPUT_FILE = "failed_jobs_diagnostics.json"

def get_headers():
    return {"Authorization": f"Bearer {TOKEN}"}

def get_all_jobs():
    """Fetch all jobs from the server"""
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/jobs",
            headers=get_headers(),
            params={"per_page": 100},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
        return None

def get_job_details(job_id):
    """Fetch detailed information for a specific job"""
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/jobs/{job_id}",
            headers=get_headers(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching job {job_id}: {e}")
        return None

def get_server_health():
    """Check server health and version"""
    try:
        response = requests.get(f"{API_BASE}/api/v1/system/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️  Error checking server health: {e}")
        return None

def analyze_failed_jobs():
    """Main diagnostic function"""
    print("🔍 VideoAnnotator Job Failure Diagnostics")
    print("=" * 60)
    print(f"Server: {API_BASE}")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Check server health
    print("📡 Checking server health...")
    health = get_server_health()
    if health:
        print(f"✅ Server version: {health.get('version', 'Unknown')}")
        print(f"✅ Status: {health.get('status', 'Unknown')}")
    else:
        print("❌ Cannot connect to server")
        return
    print()

    # Get all jobs
    print("📋 Fetching all jobs...")
    jobs_response = get_all_jobs()
    if not jobs_response:
        print("❌ Failed to fetch jobs")
        return

    jobs = jobs_response.get('jobs', [])
    print(f"✅ Found {len(jobs)} total jobs")
    print()

    # Filter failed jobs
    failed_jobs = [job for job in jobs if job.get('status') in ['failed', 'error', 'cancelled']]
    completed_jobs = [job for job in jobs if job.get('status') == 'completed']
    pending_jobs = [job for job in jobs if job.get('status') in ['pending', 'processing']]

    print(f"📊 Job Status Summary:")
    print(f"   ✅ Completed: {len(completed_jobs)}")
    print(f"   ⏳ Pending/Processing: {len(pending_jobs)}")
    print(f"   ❌ Failed: {len(failed_jobs)}")
    print()

    if not failed_jobs:
        print("🎉 No failed jobs found!")
        return

    # Detailed analysis of failed jobs
    print(f"🔍 Analyzing {len(failed_jobs)} failed jobs...")
    print("=" * 60)

    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "server": {
            "url": API_BASE,
            "version": health.get('version') if health else None,
            "status": health.get('status') if health else None
        },
        "summary": {
            "total_jobs": len(jobs),
            "completed": len(completed_jobs),
            "pending": len(pending_jobs),
            "failed": len(failed_jobs)
        },
        "failed_jobs": []
    }

    for i, job in enumerate(failed_jobs, 1):
        job_id = job.get('id')
        print(f"\n🔸 Failed Job {i}/{len(failed_jobs)}: {job_id}")
        print(f"   Status: {job.get('status')}")
        print(f"   Created: {job.get('created_at', 'Unknown')}")
        
        # Get detailed information
        details = get_job_details(job_id)
        
        job_info = {
            "job_id": job_id,
            "status": job.get('status'),
            "created_at": job.get('created_at'),
            "updated_at": job.get('updated_at'),
            "video_filename": job.get('video_filename'),
            "pipelines": job.get('pipelines', []),
            "error_message": job.get('error_message') or details.get('error_message') if details else None,
            "details": details
        }
        
        if job_info['pipelines']:
            print(f"   Pipelines: {', '.join(job_info['pipelines'])}")
        
        if job_info['error_message']:
            print(f"   ❌ Error: {job_info['error_message']}")
        else:
            print(f"   ⚠️  No error message available")
        
        diagnostics["failed_jobs"].append(job_info)

    # Save diagnostics to file
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"✅ Diagnostics saved to: {output_path.absolute()}")
    print()
    print("📧 Please share this file with the server team along with:")
    print("   1. Server logs from the time period of failed jobs")
    print("   2. Server configuration (anonymized if needed)")
    print("   3. Any console output from the server during job processing")
    print()

    # Common failure patterns
    print("🔍 Common Failure Patterns to Check:")
    print("   1. Missing dependencies (CUDA, ffmpeg, model files)")
    print("   2. Insufficient resources (disk space, memory, GPU)")
    print("   3. Permission issues (file access, directory write)")
    print("   4. Network issues (model downloads, external APIs)")
    print("   5. Configuration errors (invalid parameters, paths)")
    print()

if __name__ == "__main__":
    try:
        analyze_failed_jobs()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
