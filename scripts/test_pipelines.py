#!/usr/bin/env python3
"""
Full pipeline test execution script for VideoAnnotator.
Runs complete pipeline tests with real models and data.
Target: <15 minutes execution time.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_pipeline_tests():
    """Run the full pipeline test tier."""
    print("🔬 Running VideoAnnotator Pipeline Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test command for pipeline tests
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/pipelines/',
        '-v',
        '--tb=short',
        '--timeout=600',  # 10 minute timeout per test
        '--maxfail=3',    # Pipeline tests are expensive, fail fast
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    duration = time.time() - start_time
    
    print()
    print("=" * 50)
    if result.returncode == 0:
        print(f"✅ Pipeline tests PASSED in {duration:.1f} seconds")
    else:
        print(f"❌ Pipeline tests FAILED in {duration:.1f} seconds")
        print(f"Exit code: {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_pipeline_tests())