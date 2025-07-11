"""
VideoAnnotator Version Information and Metadata
"""

import platform
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json

# VideoAnnotator Version
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
__release_date__ = "2025-07-10"
__author__ = "VideoAnnotator Team"
__license__ = "MIT"

# Build information
__build_date__ = datetime.now().isoformat()
__git_commit__ = None  # Will be populated by CI/CD if available

def get_version_info() -> Dict[str, Any]:
    """Get comprehensive version information."""
    
    # Try to get git information if available
    git_info = get_git_info()
    
    # Get system information
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": platform.node()
    }
    
    # Get dependency versions
    dependencies = get_dependency_versions()
    
    return {
        "videoannotator": {
            "version": __version__,
            "version_info": __version_info__,
            "release_date": __release_date__,
            "build_date": __build_date__,
            "author": __author__,
            "license": __license__,
            "git": git_info
        },
        "system": system_info,
        "dependencies": dependencies,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "timezone": datetime.now().astimezone().tzinfo.tzname(None)
        }
    }

def get_git_info() -> Optional[Dict[str, str]]:
    """Get git repository information if available."""
    try:
        import subprocess
        
        # Try to get git commit hash
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except:
            commit_hash = None
        
        # Try to get git branch
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except:
            branch = None
        
        # Try to get git status (is repo clean?)
        try:
            status_output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            is_clean = len(status_output) == 0
        except:
            is_clean = None
        
        # Try to get git remote URL
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except:
            remote_url = None
        
        if any([commit_hash, branch, remote_url]):
            return {
                "commit_hash": commit_hash,
                "branch": branch,
                "is_clean": is_clean,
                "remote_url": remote_url
            }
    except ImportError:
        pass
    
    return None

def get_dependency_versions() -> Dict[str, str]:
    """Get versions of key dependencies."""
    dependencies = {}
    
    # Core dependencies
    dependency_modules = [
        ("opencv-python", "cv2", "__version__"),
        ("ultralytics", "ultralytics", "__version__"),
        ("pydantic", "pydantic", "VERSION"),
        ("numpy", "numpy", "__version__"),
        ("scenedetect", "scenedetect", "__version__"),
        ("torch", "torch", "__version__"),
        ("torchvision", "torchvision", "__version__"),
        ("pillow", "PIL", "__version__"),
        ("matplotlib", "matplotlib", "__version__"),
        ("pandas", "pandas", "__version__"),
        ("tqdm", "tqdm", "__version__"),
        ("ffmpeg-python", "ffmpeg", "__version__"),
    ]
    
    for package_name, module_name, version_attr in dependency_modules:
        try:
            module = __import__(module_name)
            if hasattr(module, version_attr):
                version = getattr(module, version_attr)
                dependencies[package_name] = str(version)
            else:
                dependencies[package_name] = "unknown"
        except ImportError:
            dependencies[package_name] = "not_installed"
        except Exception as e:
            dependencies[package_name] = f"error: {str(e)}"
    
    return dependencies

def get_model_info(model_name: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a specific model."""
    model_info = {
        "model_name": model_name,
        "model_path": model_path,
        "loaded_at": datetime.now().isoformat()
    }
    
    # Add model-specific information based on type
    if "yolo" in model_name.lower():
        model_info["model_type"] = "YOLO"
        model_info["framework"] = "Ultralytics"
        
        # Try to get model file size and modification date
        if model_path and Path(model_path).exists():
            model_file = Path(model_path)
            model_info["file_size"] = model_file.stat().st_size
            model_info["file_modified"] = datetime.fromtimestamp(
                model_file.stat().st_mtime
            ).isoformat()
    
    elif "clip" in model_name.lower():
        model_info["model_type"] = "CLIP"
        model_info["framework"] = "OpenAI"
    
    elif "whisper" in model_name.lower():
        model_info["model_type"] = "Whisper"
        model_info["framework"] = "OpenAI"
    
    else:
        model_info["model_type"] = "unknown"
    
    return model_info

def create_annotation_metadata(
    pipeline_name: str,
    model_info: Optional[Dict[str, Any]] = None,
    processing_params: Optional[Dict[str, Any]] = None,
    video_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create comprehensive metadata for annotations."""
    
    # Get the version info and use it directly as the base
    metadata = get_version_info()
    
    # Add pipeline information
    metadata["pipeline"] = {
        "name": pipeline_name,
        "processing_timestamp": datetime.now().isoformat(),
        "processing_params": processing_params or {}
    }
    
    if model_info:
        metadata["model"] = model_info
    
    if video_metadata:
        metadata["video"] = video_metadata
    
    return metadata

def save_version_info(output_path: str) -> None:
    """Save version information to a JSON file."""
    version_info = get_version_info()
    
    with open(output_path, 'w') as f:
        json.dump(version_info, f, indent=2)

def print_version_info() -> None:
    """Print version information to console."""
    version_info = get_version_info()
    
    print(f"VideoAnnotator v{__version__}")
    print(f"Release Date: {__release_date__}")
    print(f"Build Date: {__build_date__}")
    print()
    
    if version_info["videoannotator"]["git"]:
        git = version_info["videoannotator"]["git"]
        print("Git Information:")
        if git["commit_hash"]:
            print(f"  Commit: {git['commit_hash'][:8]}")
        if git["branch"]:
            print(f"  Branch: {git['branch']}")
        if git["is_clean"] is not None:
            print(f"  Clean: {git['is_clean']}")
        print()
    
    print("System Information:")
    system = version_info["system"]
    print(f"  Platform: {system['platform']}")
    print(f"  Python: {system['python_version'].split()[0]}")
    print(f"  Architecture: {system['architecture']}")
    print()
    
    print("Key Dependencies:")
    deps = version_info["dependencies"]
    key_deps = ["opencv-python", "ultralytics", "pydantic", "torch", "numpy"]
    for dep in key_deps:
        if dep in deps:
            status = deps[dep]
            if status == "not_installed":
                status = "❌ Not installed"
            elif status.startswith("error"):
                status = f"⚠️  {status}"
            else:
                status = f"✅ v{status}"
            print(f"  {dep}: {status}")

if __name__ == "__main__":
    print_version_info()
