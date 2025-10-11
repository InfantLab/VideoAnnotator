"""Quick script to load a job from storage and print full traceback on failure.

Run with: uv run python scripts/debug_load_job.py <job_id>
"""

import sys
import traceback
from pathlib import Path

# Ensure package import works from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.database import get_storage_backend

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "52325a4e-71e8-4a22-b934-0c4836fd746e"

print(f"Attempting to load job: {JOB_ID}")
storage = get_storage_backend()
try:
    job = storage.load_job_metadata(JOB_ID)
    print("Loaded job OK:", getattr(job, "job_id", None))
except Exception:
    print("Exception occurred while loading job:")
    traceback.print_exc()
    raise
