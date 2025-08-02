#!/usr/bin/env python
"""
OpenFace 3.0 Installation and Testing Script

This script helps with installing and testing OpenFace 3.0 integration
in the VideoAnnotator pipeline.
"""

import sys
import subprocess
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_system_requirements():
    """Check system requirements for OpenFace 3.0."""
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if not (3, 8) <= (python_version.major, python_version.minor) <= (3, 11):
        logger.warning("OpenFace 3.0 requires Python 3.8-3.11. Current version may not be compatible.")
    
    # Check required packages
    required_packages = ['opencv-python', 'numpy', 'scipy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.warning(f"✗ {package} is missing")
            missing_packages.append(package)
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available - will use CPU")
    except ImportError:
        logger.warning("PyTorch not installed - CUDA check skipped")
    
    return missing_packages


def install_dependencies():
    """Install OpenFace 3.0 dependencies."""
    logger.info("Installing OpenFace 3.0 dependencies...")
    
    dependencies = [
        'opencv-python>=4.5.0',
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'dlib>=19.24.0',
        'cmake>=3.16.0'
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            logger.info(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {dep}: {e}")
            return False
    
    return True


def test_openface3_import():
    """Test if OpenFace 3.0 can be imported."""
    logger.info("Testing OpenFace 3.0 import...")
    
    try:
        import openface3
        logger.info(f"✓ OpenFace 3.0 imported successfully (version: {openface3.__version__})")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import OpenFace 3.0: {e}")
        logger.info("OpenFace 3.0 is not installed. Please install it manually:")
        logger.info("1. git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git")
        logger.info("2. cd OpenFace-3.0")
        logger.info("3. mkdir build && cd build")
        logger.info("4. cmake -D CMAKE_BUILD_TYPE=RELEASE ..")
        logger.info("5. make -j$(nproc)")
        logger.info("6. cd ../python && pip install -e .")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error importing OpenFace 3.0: {e}")
        traceback.print_exc()
        return False


def test_openface3_components():
    """Test OpenFace 3.0 components."""
    logger.info("Testing OpenFace 3.0 components...")
    
    try:
        import openface3
        
        # Test face detector
        logger.info("Testing face detector...")
        detector = openface3.FaceDetector()
        logger.info("✓ Face detector initialized")
        
        # Test landmark detector
        logger.info("Testing landmark detector...")
        landmark_detector = openface3.LandmarkDetector()
        logger.info("✓ Landmark detector initialized")
        
        # Test action unit analyzer
        logger.info("Testing action unit analyzer...")
        au_analyzer = openface3.ActionUnitAnalyzer()
        logger.info("✓ Action unit analyzer initialized")
        
        # Test head pose estimator
        logger.info("Testing head pose estimator...")
        pose_estimator = openface3.HeadPoseEstimator()
        logger.info("✓ Head pose estimator initialized")
        
        # Test gaze estimator
        logger.info("Testing gaze estimator...")
        gaze_estimator = openface3.GazeEstimator()
        logger.info("✓ Gaze estimator initialized")
        
        logger.info("✓ All OpenFace 3.0 components working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing OpenFace 3.0 components: {e}")
        traceback.print_exc()
        return False


def test_openface3_pipeline():
    """Test the OpenFace 3.0 pipeline integration."""
    logger.info("Testing OpenFace 3.0 pipeline integration...")
    
    try:
        # Add the VideoAnnotator src to path
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        
        from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline
        
        # Test pipeline initialization
        config = {
            'detection_confidence': 0.7,
            'enable_action_units': True,
            'enable_head_pose': True,
            'enable_gaze': True,
            'device': 'cpu'  # Use CPU for testing
        }
        
        pipeline = OpenFace3Pipeline(config)
        logger.info("✓ OpenFace 3.0 pipeline created")
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        logger.info(f"✓ Pipeline info: {info}")
        
        # Test initialization
        pipeline.initialize()
        logger.info("✓ OpenFace 3.0 pipeline initialized successfully")
        
        logger.info("✓ OpenFace 3.0 pipeline integration working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing OpenFace 3.0 pipeline: {e}")
        traceback.print_exc()
        return False


def run_installation_test():
    """Run complete installation and testing process."""
    logger.info("=" * 60)
    logger.info("OpenFace 3.0 Installation and Testing")
    logger.info("=" * 60)
    
    # Step 1: Check system requirements
    missing_packages = check_system_requirements()
    if missing_packages:
        logger.info(f"Missing packages: {missing_packages}")
        install_deps = input("Install missing dependencies? (y/n): ").lower().strip()
        if install_deps == 'y':
            if not install_dependencies():
                logger.error("Failed to install dependencies. Exiting.")
                return False
    
    # Step 2: Test OpenFace 3.0 import
    if not test_openface3_import():
        logger.error("OpenFace 3.0 is not available. Please install it manually.")
        return False
    
    # Step 3: Test OpenFace 3.0 components
    if not test_openface3_components():
        logger.error("OpenFace 3.0 components are not working correctly.")
        return False
    
    # Step 4: Test pipeline integration
    if not test_openface3_pipeline():
        logger.error("OpenFace 3.0 pipeline integration failed.")
        return False
    
    logger.info("=" * 60)
    logger.info("✓ OpenFace 3.0 installation and integration successful!")
    logger.info("You can now use OpenFace 3.0 with VideoAnnotator:")
    logger.info("python main.py --config configs/openface3.yaml --video_path video.mp4")
    logger.info("=" * 60)
    
    return True


def show_usage_examples():
    """Show usage examples for OpenFace 3.0."""
    logger.info("Usage Examples:")
    logger.info("")
    logger.info("1. Process a single video with OpenFace 3.0:")
    logger.info("   python main.py --config configs/openface3.yaml \\")
    logger.info("                  --video_path video.mp4 \\")
    logger.info("                  --output_dir output/")
    logger.info("")
    logger.info("2. Batch process videos with OpenFace 3.0:")
    logger.info("   python main.py --config configs/openface3.yaml \\")
    logger.info("                  --batch_dir videos/ \\")
    logger.info("                  --output_dir annotations/")
    logger.info("")
    logger.info("3. Use only face analysis with OpenFace 3.0:")
    logger.info("   python main.py --config configs/openface3.yaml \\")
    logger.info("                  --pipeline face \\")
    logger.info("                  --video_path video.mp4")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        show_usage_examples()
    else:
        success = run_installation_test()
        if success:
            show_usage_examples()
        sys.exit(0 if success else 1)
