"""
OpenFace Compatibility Patches

This module provides compatibility patches for OpenFace to work with newer versions
of dependencies like SciPy.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

def patch_scipy_compatibility():
    """
    Patch SciPy compatibility issues with OpenFace.
    
    OpenFace uses scipy.integrate.simps which was deprecated and removed
    in SciPy 1.14.0+. This function provides a compatibility layer.
    """
    try:
        import scipy.integrate
        
        # Check if simps is missing (SciPy 1.14.0+)
        if not hasattr(scipy.integrate, 'simps'):
            logger.info("Applying scipy.integrate.simps compatibility patch")
            # Use simpson instead of simps
            scipy.integrate.simps = scipy.integrate.simpson
            logger.info("Successfully patched scipy.integrate.simps")
        else:
            logger.debug("scipy.integrate.simps already available, no patch needed")
            
    except ImportError as e:
        logger.warning(f"Failed to apply scipy compatibility patch: {e}")
    except Exception as e:
        logger.error(f"Unexpected error applying scipy patch: {e}")

def check_openface_compatibility():
    """
    Check OpenFace compatibility and apply necessary patches.
    
    Returns:
        bool: True if OpenFace should be compatible, False otherwise
    """
    issues = []
    
    # Check SciPy version
    try:
        import scipy
        scipy_version = tuple(map(int, scipy.__version__.split('.')[:2]))
        if scipy_version >= (1, 14):
            patch_scipy_compatibility()
            logger.info(f"Applied compatibility patches for SciPy {scipy.__version__}")
        else:
            logger.debug(f"SciPy {scipy.__version__} is compatible, no patches needed")
    except Exception as e:
        issues.append(f"SciPy compatibility check failed: {e}")
    
    # Check OpenFace availability
    try:
        import openface
        logger.info("OpenFace package is available")
    except ImportError:
        issues.append("OpenFace package not installed")
    
    # Check model availability (this would need actual model paths)
    # For now, just log a warning
    logger.warning("Model paths need to be configured for full OpenFace functionality")
    
    if issues:
        logger.error(f"Compatibility issues found: {issues}")
        return False
    
    return True

def get_default_model_paths():
    """
    Get default model paths for OpenFace.
    
    Returns:
        dict: Dictionary with model type as key and path as value
    """
    from pathlib import Path
    
    # Common model locations
    model_paths = {
        'face_detection': None,
        'landmark_detection': None,
        'action_units': None,
    }
    
    # Check common locations
    possible_locations = [
        Path.home() / '.openface' / 'models',
        Path('models'),
        Path('/usr/local/share/openface/models'),
        Path('C:/Program Files/OpenFace/models'),  # Windows
    ]
    
    for location in possible_locations:
        if location.exists():
            logger.info(f"Found potential model directory: {location}")
            # Look for specific model files
            face_models = list(location.glob('*face*detection*.pth'))
            if face_models:
                model_paths['face_detection'] = str(face_models[0])
                
            landmark_models = list(location.glob('*landmark*.pth'))
            if landmark_models:
                model_paths['landmark_detection'] = str(landmark_models[0])
                
            au_models = list(location.glob('*action*unit*.pth'))
            if au_models:
                model_paths['action_units'] = str(au_models[0])
    
    return model_paths

def download_default_models():
    """
    Download default OpenFace models if they don't exist.
    
    This is a placeholder function - actual implementation would depend
    on OpenFace's model distribution method.
    """
    logger.warning("Model download not yet implemented")
    logger.info("Please download models manually from OpenFace repository")
    logger.info("https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/releases")
    
    return False

if __name__ == "__main__":
    # Test the compatibility patches
    logging.basicConfig(level=logging.INFO)
    
    print("Testing OpenFace compatibility patches...")
    compatible = check_openface_compatibility()
    
    if compatible:
        print("✅ OpenFace compatibility checks passed")
        
        # Test imports with patches
        try:
            from openface.landmark_detection import LandmarkDetector
            print("✅ LandmarkDetector import successful with patches")
        except Exception as e:
            print(f"❌ LandmarkDetector still failing: {e}")
    else:
        print("❌ OpenFace compatibility issues detected")
    
    # Show model paths
    model_paths = get_default_model_paths()
    print(f"\nModel paths: {model_paths}")
