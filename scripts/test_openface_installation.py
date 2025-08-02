#!/usr/bin/env python3
"""
OpenFace Installation and Compatibility Test Script

This script tests the OpenFace installation and documents any compatibility issues.
"""

import sys
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic OpenFace imports."""
    print("=" * 60)
    print("Testing Basic OpenFace Imports")
    print("=" * 60)
    
    try:
        import openface
        print("✅ Basic openface import: SUCCESS")
        print(f"   OpenFace path: {openface.__path__}")
        
        # List available submodules
        import pkgutil
        submodules = [modname for importer, modname, ispkg in 
                     pkgutil.iter_modules(openface.__path__, openface.__name__ + '.')]
        print(f"   Available submodules: {submodules}")
        
    except Exception as e:
        print(f"❌ Basic openface import: FAILED")
        print(f"   Error: {e}")
        return False
    
    return True

def test_face_detection():
    """Test face detection functionality."""
    print("\n" + "=" * 60)
    print("Testing Face Detection")
    print("=" * 60)
    
    try:
        from openface.face_detection import FaceDetector
        print("✅ FaceDetector import: SUCCESS")
        
        # Try to create detector instance
        detector = FaceDetector()
        print("✅ FaceDetector instantiation: SUCCESS")
        print(f"   Detector type: {type(detector)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FaceDetector test: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def test_landmark_detection():
    """Test landmark detection functionality."""
    print("\n" + "=" * 60)
    print("Testing Landmark Detection")
    print("=" * 60)
    
    try:
        from openface.landmark_detection import LandmarkDetector
        print("✅ LandmarkDetector import: SUCCESS")
        
        # Try to create detector instance
        detector = LandmarkDetector()
        print("✅ LandmarkDetector instantiation: SUCCESS")
        print(f"   Detector type: {type(detector)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ LandmarkDetector test: FAILED (ImportError)")
        print(f"   Error: {e}")
        
        # Check if it's the scipy.integrate.simps issue
        if 'simps' in str(e) and 'simpson' in str(e):
            print("   🔧 KNOWN ISSUE: scipy.integrate.simps deprecated")
            print("   📝 Solution: Replace 'simps' with 'simpson' in OpenFace code")
            print("   📝 Workaround: Downgrade scipy or patch the code")
        
        return False
        
    except Exception as e:
        print(f"❌ LandmarkDetector test: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def test_other_modules():
    """Test other OpenFace modules."""
    print("\n" + "=" * 60)
    print("Testing Other Modules")
    print("=" * 60)
    
    results = {}
    
    # Test model module
    try:
        from openface import model
        print("✅ openface.model import: SUCCESS")
        results['model'] = True
    except Exception as e:
        print(f"❌ openface.model import: FAILED - {e}")
        results['model'] = False
    
    # Test CLI module
    try:
        from openface import cli
        print("✅ openface.cli import: SUCCESS")
        results['cli'] = True
    except Exception as e:
        print(f"❌ openface.cli import: FAILED - {e}")
        results['cli'] = False
    
    # Test demo module
    try:
        from openface import demo
        print("✅ openface.demo import: SUCCESS")
        results['demo'] = True
    except Exception as e:
        print(f"❌ openface.demo import: FAILED - {e}")
        results['demo'] = False
    
    return results

def test_with_sample_image():
    """Test OpenFace with a sample image."""
    print("\n" + "=" * 60)
    print("Testing with Sample Image")
    print("=" * 60)
    
    try:
        import numpy as np
        import cv2
        from openface.face_detection import FaceDetector
        
        # Create a simple test image (100x100 pixels)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a simple face-like pattern
        cv2.circle(test_image, (50, 50), 30, (255, 255, 255), -1)  # Face
        cv2.circle(test_image, (40, 40), 5, (0, 0, 0), -1)        # Left eye
        cv2.circle(test_image, (60, 40), 5, (0, 0, 0), -1)        # Right eye
        cv2.rectangle(test_image, (45, 55), (55, 60), (0, 0, 0), -1)  # Mouth
        
        print("✅ Test image created")
        
        # Try face detection
        detector = FaceDetector()
        results = detector.detect(test_image)
        print(f"✅ Face detection completed: {len(results) if results else 0} faces detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample image test: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def check_environment():
    """Check the current environment."""
    print("=" * 60)
    print("Environment Information")
    print("=" * 60)
    
    import platform
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check key package versions
    packages_to_check = [
        'numpy', 'opencv-cv2', 'torch', 'torchvision', 'scipy', 
        'matplotlib', 'pillow', 'pandas', 'scikit-image'
    ]
    
    for package in packages_to_check:
        try:
            if package == 'opencv-cv2':
                import cv2
                print(f"OpenCV: {cv2.__version__}")
            elif package == 'pillow':
                import PIL
                print(f"Pillow: {PIL.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: NOT INSTALLED")
        except Exception as e:
            print(f"{package}: ERROR - {e}")

def main():
    """Main test function."""
    print("OpenFace Installation and Compatibility Test")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Run tests
    results = {
        'basic_imports': test_basic_imports(),
        'face_detection': test_face_detection(),
        'landmark_detection': test_landmark_detection(),
        'other_modules': test_other_modules(),
        'sample_image': test_with_sample_image()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    total_tests = len([k for k in results.keys() if k != 'other_modules'])
    passed_tests = sum([1 for k, v in results.items() if k != 'other_modules' and v])
    
    if 'other_modules' in results:
        other_passed = sum(results['other_modules'].values())
        other_total = len(results['other_modules'])
        total_tests += other_total
        passed_tests += other_passed
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! OpenFace is ready to use.")
    elif passed_tests > 0:
        print("⚠️ Partial success. Some functionality available but issues detected.")
    else:
        print("❌ OpenFace installation has significant issues.")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    if not results.get('landmark_detection', False):
        print("📝 LandmarkDetector failed - likely due to scipy compatibility")
        print("   Recommended fix: Patch scipy.integrate.simps usage")
        print("   or downgrade scipy to <1.12.0")
    
    if results.get('face_detection', False):
        print("✅ FaceDetector working - basic face detection available")
    
    print("\n🔗 For detailed troubleshooting, see: docs/OPENFACE3_GUIDE.md")

if __name__ == "__main__":
    main()
