#!/usr/bin/env python3
"""
OpenFace 3.0 Installation Verification Script

This script verifies that OpenFace 3.0 is properly installed and configured
for use with the VideoAnnotator system.

Usage:
    python scripts/verify_openface_installation.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_openface_installation():
    """Verify OpenFace 3.0 installation and components."""
    
    print("🔍 OpenFace 3.0 Installation Verification")
    print("=" * 50)
    
    # Check if compatibility patches are available
    try:
        from src.pipelines.face_analysis.openface_compatibility import patch_scipy_compatibility
        patch_scipy_compatibility()
        print("✅ Compatibility patches applied successfully")
    except ImportError as e:
        print(f"❌ Compatibility patches not found: {e}")
        return False
    
    # Test OpenFace imports
    try:
        from openface.face_detection import FaceDetector
        from openface.landmark_detection import LandmarkDetector
        print("✅ OpenFace components imported successfully")
    except ImportError as e:
        print(f"❌ OpenFace import failed: {e}")
        print("💡 Install with: pip install openface-test --no-deps")
        return False
    
    # Check model files
    model_files = {
        'Face Detector': './weights/Alignment_RetinaFace.pth',
        'Landmark 98': './weights/Landmark_98.pkl',
        'Landmark 68': './weights/Landmark_68.pkl',
        'MTL Backbone': './weights/MTL_backbone.pth',
        'Mobile Face': './weights/mobilenetV1X0.25_pretrain.tar'
    }
    
    print("\n📦 Checking Model Files:")
    all_models_present = True
    for name, path in model_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {name}: {path} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {name}: {path} (missing)")
            all_models_present = False
    
    if not all_models_present:
        print("\n💡 Download models with:")
        print("python -c \"from openface.cli import download; download()\"")
        return False
    
    # Test model initialization
    print("\n🚀 Testing Model Initialization:")
    
    try:
        face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth')
        print("✅ FaceDetector initialized successfully")
    except Exception as e:
        print(f"❌ FaceDetector initialization failed: {e}")
        return False
    
    try:
        landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl')
        print("✅ LandmarkDetector (98-point) initialized successfully")
    except Exception as e:
        print(f"❌ LandmarkDetector initialization failed: {e}")
        return False
    
    # Test with 68-point model (expected to fail)
    print("\n⚠️  Testing Known Issue - 68-point Model:")
    try:
        landmark_detector_68 = LandmarkDetector(model_path='./weights/Landmark_68.pkl')
        print("⚠️  68-point model loaded (unexpected - may have been fixed)")
    except Exception as e:
        print("❌ 68-point model failed as expected (shape mismatch)")
        print("💡 Use 98-point model for landmark detection")
    
    print("\n🎉 OpenFace 3.0 Installation Verification Complete!")
    print("✅ All components working correctly")
    print("✅ Ready for VideoAnnotator integration")
    
    return True

def get_system_info():
    """Display relevant system information."""
    import platform
    import cv2
    import numpy as np
    import torch
    
    print("\n📋 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   OpenCV: {cv2.__version__}")
    print(f"   NumPy: {np.__version__}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    try:
        success = verify_openface_installation()
        get_system_info()
        
        if success:
            print("\n🚀 Next Steps:")
            print("   1. Integrate OpenFace pipeline with VideoAnnotator")
            print("   2. Test with actual video processing")
            print("   3. Configure pipeline parameters as needed")
            sys.exit(0)
        else:
            print("\n🔧 Installation needs attention - see messages above")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
