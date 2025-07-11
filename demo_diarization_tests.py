"""
Demonstration script for running PyAnnote diarization pipeline tests.

This script shows how to run the diarization tests and provides
information about the testing framework.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Main demonstration function."""
    
    print("🎯 PyAnnote Diarization Pipeline Testing")
    print("=" * 50)
    
    print("\n📋 Available Test Commands:")
    print("1. Run all diarization tests:")
    print("   python -m pytest tests/test_pipelines.py::TestDiarizationPipeline -v")
    
    print("\n2. Run basic unit tests (no external dependencies):")
    print("   python -m pytest tests/test_pipelines.py::TestDiarizationPipeline -v -m 'not integration'")
    
    print("\n3. Run integration tests (requires HuggingFace token):")
    print("   TEST_INTEGRATION=1 HUGGINGFACE_TOKEN=your_token python -m pytest tests/test_pipelines.py::TestDiarizationPipelineIntegration -v")
    
    print("\n4. Run with coverage:")
    print("   python -m pytest tests/test_pipelines.py::TestDiarizationPipeline --cov=src.pipelines.audio_processing")
    
    print("\n📁 Test Structure:")
    print("   tests/test_pipelines.py")
    print("   ├── TestDiarizationPipeline (unit tests)")
    print("   │   ├── Configuration tests")
    print("   │   ├── Initialization tests") 
    print("   │   ├── Processing tests (mocked)")
    print("   │   └── Error handling tests")
    print("   └── TestDiarizationPipelineIntegration (integration tests)")
    print("       └── Real PyAnnote model tests")
    
    print("\n🧪 Running Basic Tests Now...")
    print("-" * 30)
    
    # Run basic unit tests
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_pipelines.py::TestDiarizationPipeline", 
            "-v", "-x"  # Stop on first failure
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All basic diarization tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
    
    print("\n📚 Next Steps:")
    print("1. Set HUGGINGFACE_TOKEN environment variable")
    print("2. Install pyannote.audio if not already installed")
    print("3. Run integration tests with real data")
    print("4. Add your own test cases for specific scenarios")
    
    print("\n🔗 Useful Links:")
    print("- PyAnnote: https://github.com/pyannote/pyannote-audio")
    print("- HuggingFace Token: https://huggingface.co/settings/tokens")
    print("- Model Page: https://huggingface.co/pyannote/speaker-diarization-3.1")


if __name__ == "__main__":
    main()
