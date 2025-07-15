"""
Manual test runner for batch validation tests.
Use this to run specific test classes or individual tests.

Usage examples:
    python run_batch_tests.py                          # Run all tests
    python run_batch_tests.py TestBatchTypesValidation # Run specific test class
    python run_batch_tests.py test_batch_job_creation  # Run specific test method
"""

import sys
import subprocess
from pathlib import Path

def run_pytest(test_filter=None):
    """Run pytest with optional filter."""
    cmd = ["python", "-m", "pytest", "tests/test_batch_validation.py", "-v"]
    
    if test_filter:
        cmd.extend(["-k", test_filter])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        test_filter = sys.argv[1]
        print(f"Running tests matching: {test_filter}")
    else:
        test_filter = None
        print("Running all batch validation tests")
    
    success = run_pytest(test_filter)
    
    if success:
        print("\n🎉 Tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
