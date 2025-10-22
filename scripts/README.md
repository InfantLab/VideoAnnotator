# VideoAnnotator Scripts

Utility scripts for VideoAnnotator development, testing, and validation.

## Installation Verification

### `verify_installation.py`

Progressive environment validation script for users and JOSS reviewers to verify their VideoAnnotator installation before use.

**Purpose**: Validate that all critical components are properly installed and configured.

**Usage**:
```bash
# Full verification (includes video processing test)
python scripts/verify_installation.py

# Skip video processing test (faster)
python scripts/verify_installation.py --skip-video-test

# Verbose output with detailed diagnostics
python scripts/verify_installation.py --verbose
```

**Checks Performed**:

1. **Python Version** (CRITICAL)
   - Validates Python >= 3.10
   - Displays current version

2. **FFmpeg Availability** (CRITICAL)
   - Checks `ffmpeg` command availability
   - Validates FFmpeg can execute

3. **VideoAnnotator Import** (CRITICAL)
   - Verifies package is installed/importable
   - Detects version number

4. **Database Write Access** (CRITICAL)
   - Creates test SQLite database
   - Validates write permissions

5. **GPU Availability** (OPTIONAL)
   - Detects CUDA-capable GPUs
   - Reports GPU names and count
   - Non-critical (CPU fallback available)

6. **Video Processing** (OPTIONAL)
   - Generates test video with FFmpeg
   - Reads video with OpenCV
   - Validates full pipeline functionality
   - Can be skipped with `--skip-video-test`

**Exit Codes**:
- `0`: All checks passed
- `1`: Critical failure (must be fixed)
- `2`: Warnings only (non-critical issues)

**Platform Support**:
- Linux (native)
- macOS
- Windows
- WSL2 (auto-detected)

**Output Format**:
- ASCII-safe (no emoji for Windows compatibility)
- Actionable error messages with fix suggestions
- Clear success/failure indicators
- Summary with next steps

**Example Output**:
```
================================================================================
VideoAnnotator Installation Verification
================================================================================

[INFO] Platform: Windows WSL2 (x86_64, CPython)

[CHECK] Python version... [OK] 3.12.3
[CHECK] FFmpeg availability... [OK] FFmpeg 6.0
[CHECK] VideoAnnotator package import... [OK] v1.3.0-dev
[CHECK] Database write access... [OK] Writable
[CHECK] GPU availability (optional)... [OK] 1 GPU(s) - NVIDIA RTX 3080
[CHECK] Sample video processing... [OK] Processed 5 frames

================================================================================
Verification Summary
================================================================================

Checks passed: 6/6

[OK] Installation complete! You can now:
  - Run examples: python examples/basic_video_processing.py
  - Start API server: videoannotator server
  - View pipelines: videoannotator pipelines --detailed
```

**Testing**:

Comprehensive test suite available in `tests/unit/scripts/test_verify_installation.py`:
- 30 tests covering all check scenarios
- Mock-based testing for subprocess, imports, file operations
- Platform detection tests for all OS types
- Exit code verification

Run tests:
```bash
uv run pytest tests/unit/scripts/ -v
```

**For JOSS Reviewers**:

This script provides automated verification of installation requirements listed in our paper. Running this before evaluation ensures:
- All dependencies are correctly installed
- Platform-specific requirements are met
- GPU acceleration (if available) is properly configured
- Basic functionality is working

**Development Notes**:

The `scripts/` directory is a proper Python package (contains `__init__.py`) to enable clean imports in tests. This allows:
```python
from scripts.verify_installation import InstallationVerifier
```

Rather than fragile sys.path manipulation.

## Other Scripts

Additional development and testing scripts documented below as they are added.
