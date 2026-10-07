# VideoAnnotator Troubleshooting Guide

This guide helps you resolve common installation and runtime issues. Most issues can be resolved in 5-10 minutes.

> **Quick Links**: [The start-up program](#the-start-up-program) | [Common Issues](#common-issues) | [GPU/CUDA](#gpu-and-cuda-issues) | [Database](#database-issues) | [Network](#network-issues) | [Diagnostics](#diagnostic-commands)

## The start-up program

`videoannotator-start` (and the **Start VideoAnnotator** shortcut) says what went wrong in one line,
with the next step. Each message, and more about what to do:

### Install Docker or Podman

> VideoAnnotator needs Docker Desktop or Podman Desktop. Install one (see ...), then run this again.

Neither is installed (or neither is on your `PATH`). Install
[Docker Desktop](https://www.docker.com/products/docker-desktop/) or
[Podman Desktop](https://podman-desktop.io/), start it once so it can finish setting up, then start
VideoAnnotator again.

### Docker isn't running

> Docker Desktop isn't running. Start it, wait until it says it's running, then run this again.

Open Docker Desktop and wait until its status says it is running (the whale icon stops animating).

> Docker isn't running. Start it with: sudo systemctl start docker, then run this again.

Linux, Docker Engine: the service is stopped. `sudo systemctl enable --now docker` also starts it at
every boot.

> You don't have permission to use Docker yet. Run: sudo usermod -aG docker $USER, log out and back in, then run this again. (Or use Podman, which needs no permission.)

Linux: only members of the `docker` group may use Docker Engine. Logging out and back in is what
makes the new group count. Podman needs no such permission.

### Podman's machine

> Starting Podman's virtual machine (first time takes a minute)...

On Windows and macOS, Podman runs containers in a small virtual machine; the start-up program starts
it (and creates it, the first time). Nothing to do. If it fails, open Podman Desktop and start the
machine there.

### Port in use

> Something else is using port 18011. Close it, or run: videoannotator-start --port 18012

Another program (perhaps another VideoAnnotator, started without the start-up program) is using the
port. Close it, or start with `--port 18012`; the start-up program remembers the port.

### Download failed

> Couldn't download VideoAnnotator. Check your internet connection and run this again.

The first start downloads about 1 GB from `ghcr.io`. Behind a proxy, set it in Docker Desktop
(Settings, Resources, Proxies) or for Podman (`HTTPS_PROXY` in the machine).

### Memory

> VideoAnnotator ran out of memory. Give Docker/Podman more (see ...), then run this again.

On Windows and macOS, containers get a share of your computer's memory. Give them at least 8 GB
(16 GB for face and speech pipelines): Docker Desktop, Settings, Resources; Podman Desktop,
Settings, Resources, or `podman machine set --memory 8192` with the machine stopped. On Windows with
WSL 2, Docker Desktop's memory is set in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=8GB`).

### GPU

> Running without the GPU: Docker can't use it yet. To enable it, see ...

You have an NVIDIA GPU, but the engine can't pass it to VideoAnnotator yet; it runs on the CPU
meanwhile. What each engine needs is in the
[installation guide](INSTALLATION.md#using-the-gpu). If a start with the GPU fails, the start-up
program starts once more without it and says so.

### Docker Desktop file sharing

> Docker Desktop can't see /opt/data yet. Add it in Docker Desktop's Settings, Resources, File sharing (see ...), then run this again.

macOS: Docker Desktop shares `/Users`, `/Volumes`, `/private` and `/tmp` by default. To share a
folder elsewhere, add it under Settings, Resources, File sharing, then apply and restart.

### Already running

> VideoAnnotator is already running.

Started twice: the browser opens on the one already running. Nothing to do.

### Shared folders

> Couldn't find E:\Data (an unplugged drive?), so it isn't shared this time.

A shared folder wasn't there at start. Plug the drive in and start again; it is still remembered.
To forget it: `videoannotator-start unshare`.

> This shares everything in your home folder, including documents unrelated to your research. Share it anyway? [y/N]

Sharing a very broad folder lets VideoAnnotator read much more than your videos. Choose the folder
your videos are in instead, unless you really mean it.

> That folder is inside your results folder, which VideoAnnotator can already read.

Results are already readable (and writable); there's no need to share them.

### Running videos and restarts

> 2 videos are being processed. [W]ait for them, or [r]estart now (they'll be marked failed and can be retried)?

Sharing a folder, stopping sharing one, and updating all restart VideoAnnotator. Waiting is the
default. Videos still queued keep their place either way.

### Anything else

> VideoAnnotator couldn't start. Run: videoannotator-start logs

The lines after it are the engine's own words. `videoannotator-start logs` shows VideoAnnotator's
log; include both when you [report a problem](https://github.com/InfantLab/VideoAnnotator/issues).

## Common Issues

### FFmpeg Not Found

**Symptoms**:
- Error: `ffmpeg: command not found`
- Video processing fails with "FFmpeg not available"

**Solution**:

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS**:
```bash
brew install ffmpeg
```

**Windows**:
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit `Path` variable
   - Add `C:\ffmpeg\bin`
4. Restart terminal and verify: `ffmpeg -version`

**Verification**:
```bash
ffmpeg -version
# Should show FFmpeg version info
```

---

### Python Version Mismatch

**Symptoms**:
- Error: `Python 3.12+ required`
- Import errors with type hints

**Solution**:

Check your Python version:
```bash
python --version
# or
python3 --version
```

If version is < 3.12, install Python 3.12:

**Linux (Ubuntu)**:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**macOS**:
```bash
brew install python@3.12
```

**Windows**:
Download from [python.org](https://www.python.org/downloads/) and install Python 3.12+

**With uv (recommended)**:
```bash
# uv automatically uses correct Python version
uv sync
```

---

### Import Errors - Module Not Found

**Symptoms**:
- `ModuleNotFoundError: No module named 'videoannotator'`
- `ImportError: cannot import name 'X' from 'Y'`

**Solution**:

**1. Ensure dependencies are installed**:
```bash
# Sync all dependencies
uv sync

# If that fails, try fresh install
rm -rf .venv
uv sync
```

**2. Verify installation**:
```bash
uv run python -c "import videoannotator; print(videoannotator.__version__)"
```

**3. Check you're using uv run**:
```bash
# Wrong - uses system Python
python my_script.py

# Correct - uses project environment
uv run python my_script.py
```

**4. Editable install issues**:
```bash
# Reinstall in editable mode
uv pip install -e .
```

---

### Port Already in Use

**Symptoms**:
- Error: `Address already in use: 18011`
- API server won't start

**Solution**:

**Option 1 - Find and kill process**:

Linux/Mac:
```bash
# Find process using port 18011
lsof -i :18011

# Kill the process
kill -9 <PID>
```

Windows (PowerShell):
```powershell
# Find process
netstat -ano | findstr :18011

# Kill process
taskkill /PID <PID> /F
```

**Option 2 - Use different port**:
```bash
# Set custom port
export API_PORT=18012
uv run videoannotator server --host 0.0.0.0 --port 18012

# Or in code
uvicorn videoannotator.api.main:app --port 18012
```

---

### Out of Disk Space

**Symptoms**:
- Error: `No space left on device`
- Database writes fail
- Model downloads incomplete

**Solution**:

**1. Check disk usage**:
```bash
df -h
```

**2. Clean up**:
```bash
# Remove old logs (folder printed at server start; Linux default shown)
rm -f ~/.local/state/videoannotator/logs/*.log

# Clean test artifacts
rm -rf test_storage/

# Remove model weights (re-downloaded when needed). Where they are, and
# how much space each source takes:
videoannotator diagnose models

# Clean uv cache
uv cache clean
```

**3. Move storage to larger disk**:
```bash
# Set custom storage path
export STORAGE_BASE_PATH=/path/to/larger/disk/storage
```

---

### Permission Denied Errors

**Symptoms**:
- `PermissionError: [Errno 13] Permission denied`
- Cannot write to database
- Cannot create directories

**Solution**:

**1. Check file ownership**:
```bash
ls -la custom_storage/
ls -la tokens/
```

**2. Fix ownership** (Linux/Mac):
```bash
# Fix ownership of storage directories
sudo chown -R $USER:$USER custom_storage/
sudo chown -R $USER:$USER tokens/
sudo chown -R $USER:$USER logs/
```

**3. Fix permissions**:
```bash
chmod -R u+rw custom_storage/
chmod -R u+rw tokens/
chmod -R u+rw logs/
```

**Windows**: Run terminal as Administrator if permission issues persist.

---

### Installation Verification Fails

**Symptoms**:
- `videoannotator diagnose` reports failures

**Solution**:

Run with verbose output to see specific issues:
```bash
uv run videoannotator diagnose
```

Common fixes:
- **Python check fails**: Install Python 3.12+
- **FFmpeg check fails**: Install FFmpeg (see above)
- **Import check fails**: Run `uv sync`
- **Database check fails**: Fix permissions on `custom_storage/`
- **GPU check fails**: See GPU/CUDA section below
- **Video test fails**: Use `--skip-video-test` if no video needed

---

### The dev container is slow, freezes or runs out of memory

**Symptoms**:
- Tests, imports and model loading are far slower than on Linux or macOS (Windows), with
  `MsMpEng.exe` (Defender) high in Task Manager, or the container printed
  `WARNING: this workspace is a Windows folder mounted into the container` when it started
- A command ends with `Killed`, or exit code 137
- The whole machine stops responding (Windows), often just after waking from sleep with the
  container still running

**Slow on Windows**: the project is in a Windows folder (e.g. `C:\Users\you\code\VideoAnnotator`)
opened with **Reopen in Container**, so every file the container reads crosses the Windows–WSL file
bridge and Defender scans it. Open the project from a container volume instead: in VS Code,
**Dev Containers: Clone Repository in Container Volume…**, then
`https://github.com/InfantLab/VideoAnnotator`. Commit and push anything in your Windows clone
first; the volume clone starts from GitHub. Optionally, if you administer the machine, put code on a
[Dev Drive](https://learn.microsoft.com/windows/dev-drive/) or exclude your code folder from
Defender's real-time scanning.

**`Killed` or exit 137**: the process ran out of memory. The container is capped at 12 GB
(`--memory=12g` in `.devcontainer/devcontainer.json`), and Docker Desktop's VM may be smaller: by
default it gets half your RAM, so 8 GB on a 16 GB machine. Normal work stays well inside 8 GB (about
2 GB idle, 5.4 GB for the full test suite, 6.5 GB for a six-pipeline job), so this usually means a
long video, several jobs at once, or other containers running. Give the VM more memory (Docker
Desktop → Settings → Resources, or `memory=` in `%USERPROFILE%\.wslconfig` then `wsl --shutdown`),
and raise `--memory` in `devcontainer.json` if the container's own cap was the limit. A full cap
while installing packages is not this: that is page cache, which is reclaimed.

**Freezes on Windows**: not fully explained. In the one case examined (2026-10-02) the laptop had
slept overnight with the container running, and within minutes of waking the Docker VM locked up
while handing memory back to Windows (a `soft lockup` in `page_reporting_process`). Until that is
understood, stop the container (or quit Docker Desktop) before the machine sleeps. If Docker
Desktop's **Resource Saver** mode is on (Settings → Resources), try turning it off: it has been
linked to WSL hangs after sleep. If your machine freezes anyway, please open an issue with the time
it happened and what was running; we are still collecting cases.

**Model weights and the Python environment** live in Docker named volumes, `videoannotator-models`
and `videoannotator-venv`, whichever way you open the project. They survive rebuilding the
container and `docker system prune --volumes`, which removes only anonymous volumes (Docker 23
and later). Only `docker volume rm`, `docker volume prune --all` or resetting Docker Desktop
deletes them.

```bash
# On the host. Back up the weights to the current directory:
docker run --rm -v videoannotator-models:/models -v "$PWD":/backup alpine tar czf /backup/videoannotator-models.tgz -C /models .
# Wipe them (they download again when a pipeline needs them):
docker volume rm videoannotator-models
```

Weights already downloaded into the old `models/` folder of a Windows clone can be reused: with
the clone open in its container, `cp -a models/. /app/models/` copies them into the volume once.
The server lists such old locations when it starts.

---

## GPU and CUDA Issues

### GPU Not Detected

**Symptoms**:
- `torch.cuda.is_available()` returns `False`
- Models run slowly (CPU mode)
- Warning: "CUDA not available"

**Solution**:

**1. Verify NVIDIA GPU exists**:

Linux:
```bash
lspci | grep -i nvidia
# or
nvidia-smi
```

Windows:
```powershell
nvidia-smi
```

If `nvidia-smi` not found, install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)

**2. Check CUDA installation**:
```bash
nvcc --version
```

If not found, install CUDA Toolkit 12.4+:
- Linux: Follow [NVIDIA CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
- Windows: Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

**3. Install CUDA-enabled PyTorch**:
```bash
# Remove CPU-only PyTorch
uv remove torch torchvision torchaudio

# Install CUDA 12.4 version
uv add "torch==2.8.*+cu124" "torchvision==0.21.*+cu124" "torchaudio==2.8.*+cu124" --index-url https://download.pytorch.org/whl/cu124
```

**4. Verify GPU access**:
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

### CUDA Out of Memory

**Symptoms**:
- Error: `RuntimeError: CUDA out of memory`
- Process crashes during model inference

**Solution**:

**1. Check GPU memory**:
```bash
nvidia-smi
```

**2. Reduce batch size** in pipeline configs:
```yaml
# In config file
batch_size: 1  # Reduce from default
```

**3. Use mixed precision**:
```yaml
# Enable FP16 inference
use_fp16: true
```

**4. Clear GPU cache** between runs:
```python
import torch

torch.cuda.empty_cache()
```

**5. Upgrade GPU** if consistently running out of memory:
- Minimum: GTX 1060 6GB
- Recommended: RTX 3060 12GB or better

---

### CUDA Version Mismatch

**Symptoms**:
- Error: `CUDA version mismatch`
- PyTorch fails to initialize CUDA

**Solution**:

**1. Check installed CUDA version**:
```bash
nvidia-smi  # Look at top right corner
nvcc --version
```

**2. Match PyTorch CUDA version**:

For CUDA 12.4:
```bash
uv add "torch==2.8.*+cu124" "torchvision==0.21.*+cu124" --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.1:
```bash
uv add "torch==2.8.*+cu121" "torchvision==0.21.*+cu121" --index-url https://download.pytorch.org/whl/cu121
```

For CPU only (no GPU):
```bash
uv add torch torchvision torchaudio
```

---

### macOS GPU Support

**Note**: CUDA is not available on macOS. Apple Silicon (M1/M2/M3) uses Metal Performance Shaders (MPS), but VideoAnnotator currently runs in CPU mode on macOS.

**Solution**: Use Linux or Windows for GPU acceleration, or accept CPU-only performance on macOS.

---

## Database Issues

### Database Locked Error

**Symptoms**:
- Error: `database is locked`
- SQLite operational error

**Solution**:

**1. Check for concurrent access**:
```bash
# Find processes accessing database
lsof custom_storage/jobs.db
```

**2. Wait for long-running jobs** to complete

**3. Restart API server**:
```bash
# Stop server (Ctrl+C)
# Start fresh
uv run videoannotator server --host 0.0.0.0 --port 18011
```

**4. Use write-ahead logging** (WAL mode):
```bash
sqlite3 custom_storage/jobs.db "PRAGMA journal_mode=WAL;"
```

---

### Database Corrupted

**Symptoms**:
- Error: `database disk image is malformed`
- Cannot query jobs

**Solution**:

**1. Backup existing database**:
```bash
cp custom_storage/jobs.db custom_storage/jobs.db.backup
```

**2. Attempt SQLite recovery**:
```bash
sqlite3 custom_storage/jobs.db ".recover" | sqlite3 custom_storage/jobs_recovered.db
mv custom_storage/jobs_recovered.db custom_storage/jobs.db
```

**3. If recovery fails, start fresh**:
```bash
rm custom_storage/jobs.db
# Database will be recreated on next API start
uv run videoannotator server --host 0.0.0.0 --port 18011
```

**Prevention**: Regular backups:
```bash
# Add to cron/scheduled task
sqlite3 custom_storage/jobs.db ".backup 'backups/jobs_$(date +%Y%m%d).db'"
```

---

### Cannot Write to Database

**Symptoms**:
- Error: `attempt to write a readonly database`
- Job submission fails

**Solution**:

**1. Check permissions**:
```bash
ls -la custom_storage/
```

**2. Fix ownership and permissions**:
```bash
# Linux/Mac
sudo chown -R $USER:$USER custom_storage/
chmod u+w custom_storage/jobs.db

# Windows - run as Administrator
icacls custom_storage\jobs.db /grant %USERNAME%:F
```

**3. Check disk space**:
```bash
df -h
```

---

## Network Issues

### API Returns 401 Unauthorized

**Symptoms**:
- All API requests return 401
- Error: "Invalid or missing API key"

**Solution**:

**1. Check if authentication is enabled**:
```bash
# Check environment
echo $AUTH_REQUIRED  # Should be "true" or empty (defaults to true)
```

**2. Get your API key**:
```bash
uv run python scripts/auth/get_api_key.py
```

**3. Use API key in requests**:
```bash
export API_KEY="va_api_xxx..."

curl -X GET "http://localhost:18011/api/v1/jobs" \
  -H "X-API-Key: $API_KEY"
```

**4. Disable authentication for local testing** (not recommended for production):
```bash
uv run videoannotator server --dev --host 0.0.0.0 --port 18011
```

See [Authentication Guide](../security/authentication.md) for details.

---

### Connection Refused / Cannot Connect

**Symptoms**:
- Error: `Connection refused`
- `curl: (7) Failed to connect`

**Solution**:

**1. Verify server is running**:
```bash
# Check process
ps aux | grep "videoannotator.api.main" | grep -v grep

# Check port
netstat -an | grep 18011
# or
lsof -i :18011
```

**2. Start server if not running**:
```bash
uv run videoannotator server --host 0.0.0.0 --port 18011
```

**3. Check firewall** (if accessing remotely):

Linux:
```bash
sudo ufw allow 18011
```

Windows:
```powershell
netsh advfirewall firewall add rule name="VideoAnnotator" dir=in action=allow protocol=TCP localport=18011
```

**4. Check host binding**:

Server should bind to `0.0.0.0` for external access:
```python
uvicorn.run("videoannotator.api.main:app", host="0.0.0.0", port=18011)
```

---

### CORS Errors in Browser

**Symptoms**:
- Browser console: `CORS policy: No 'Access-Control-Allow-Origin' header`
- Fetch requests fail from web app
- API works in Postman/curl but not in browser

**Quick Solution**:
```bash
# Standard usage: Official client (port 19011) works automatically
uv run videoannotator

# Custom client development: Use dev mode (allows all CORS origins)
uv run videoannotator --dev
```

**What's Allowed by Default** (v1.3.0+):
The server automatically allows:
- `http://localhost:19011` (video-annotation-viewer - official client)
- `http://localhost:18011` (server port - same-origin requests)

**Custom Client Ports**:
If developing a custom client on a different port:

```bash
# Option 1: Set specific origin
export CORS_ORIGINS="http://localhost:YOUR_PORT"
uv run videoannotator

# Option 2: Allow all origins (development only)
uv run videoannotator --dev
```

**Production Configuration**:
```bash
# Only allow your production domain
export CORS_ORIGINS="https://viewer.example.com"
uv run videoannotator
```

**Verification**:
Check browser console for CORS headers:
```javascript
// In browser console
fetch('http://localhost:18011/health')
  .then(r => console.log(r.headers.get('access-control-allow-origin')))
```

See [CORS Guide](../security/cors.md) for detailed configuration.

---

## Diagnostic Commands

Use these commands to gather information for troubleshooting:

### System Information

```bash
# Python version
python --version

# uv version
uv --version

# FFmpeg version
ffmpeg -version

# CUDA version (if installed)
nvcc --version
nvidia-smi

# Disk space
df -h

# Memory usage
free -h  # Linux
vm_stat  # macOS
```

### VideoAnnotator Status

```bash
# Package version
uv run python -c "import videoannotator; print(videoannotator.__version__)"

# Installation verification
uv run videoannotator diagnose

# Check imports
uv run python -c "
import torch
import cv2
import whisper
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print('All imports successful')
"
```

### API Health Check

```bash
# Basic health
curl http://localhost:18011/health

# Detailed health with API key
export API_KEY="va_api_xxx..."
curl -X GET "http://localhost:18011/api/v1/system/health" \
  -H "X-API-Key: $API_KEY"

# Check specific endpoints
curl -X GET "http://localhost:18011/api/v1/pipelines" \
  -H "X-API-Key: $API_KEY"
```

### Database Status

```bash
# Check database file
ls -lh custom_storage/jobs.db

# Count jobs
sqlite3 custom_storage/jobs.db "SELECT COUNT(*) FROM jobs;"

# Recent jobs
sqlite3 custom_storage/jobs.db "SELECT id, status, created_at FROM jobs ORDER BY created_at DESC LIMIT 5;"

# Check for locks
lsof custom_storage/jobs.db
```

### Log Analysis

```bash
# View recent logs (folder printed at server start as "[INFO] Logs: ...";
# Linux default shown, /app/logs in Docker)
tail -f ~/.local/state/videoannotator/logs/api_server.log

# Search for errors
tail -50 ~/.local/state/videoannotator/logs/errors.log

# Search for specific job
grep "job_abc123" ~/.local/state/videoannotator/logs/api_server.log

# Check API request logs
grep "POST /api/v1/jobs" ~/.local/state/videoannotator/logs/api_requests.log
```

---

## Advanced Troubleshooting

### Enable Debug Logging

Set environment variable for verbose output:

```bash
uv run python api_server.py --log-level debug
```

Or modify `src/utils/logging_config.py`:
```python
logger.setLevel(logging.DEBUG)
```

### Test Individual Pipelines

Run pipelines in isolation to identify issues:

```bash
# Test OpenFace pipeline
uv run python examples/test_individual_pipelines.py --pipeline openface3_identity --video test.mp4

# Test Whisper pipeline
uv run python examples/test_individual_pipelines.py --pipeline whisper_transcription --video test.mp4
```

### Memory Profiling

Track memory usage to identify leaks:

```bash
# Install memory profiler
uv add memory-profiler

# Profile script
uv run python -m memory_profiler your_script.py
```

### Network Debugging

Capture network traffic for API issues:

```bash
# Use httpie for better debugging
uv add httpie

http GET http://localhost:18011/api/v1/jobs X-API-Key:$API_KEY

# Or curl with verbose output
curl -v -X GET "http://localhost:18011/api/v1/jobs" \
  -H "X-API-Key: $API_KEY"
```

---

## Getting Help

If you can't resolve the issue:

1. **Check GitHub Issues**: [github.com/InfantLab/VideoAnnotator/issues](https://github.com/InfantLab/VideoAnnotator/issues)

2. **Gather diagnostic information**:
   ```bash
   # Run full diagnostic
   uv run videoannotator diagnose > diagnostic.txt 2>&1

   # Include system info
   uname -a >> diagnostic.txt
   python --version >> diagnostic.txt
   nvidia-smi >> diagnostic.txt
   ```

3. **Create detailed issue**:
   - OS and version
   - Python version
   - VideoAnnotator version
   - Steps to reproduce
   - Error messages (full traceback)
   - Diagnostic output
   - What you've tried

4. **Check documentation**:
   - [Installation Guide](INSTALLATION.md)
   - [Getting Started](../usage/GETTING_STARTED.md)
   - [Security Guide](../security/README.md)
   - [Testing Guide](../testing/README.md)

---

## Common Issue Resolution Time

| Issue Category | Typical Resolution Time |
|----------------|------------------------|
| FFmpeg/Python version | 5-10 minutes |
| Import errors | 2-5 minutes |
| Port conflicts | 1-2 minutes |
| Permission errors | 3-5 minutes |
| GPU/CUDA setup | 15-30 minutes |
| Database issues | 5-10 minutes |
| Network/CORS | 5-10 minutes |
| Authentication | 2-5 minutes |

**Target**: 80% of issues resolved in <15 minutes ✅
