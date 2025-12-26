"""Test that face_openface3_embedding pipeline is available in both API and execution engines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from videoannotator.api.job_processor import JobProcessor
from videoannotator.batch.batch_orchestrator import BatchOrchestrator
from videoannotator.storage.file_backend import FileStorageBackend


def test_face_openface3_embedding():
    """Verify face_openface3_embedding is registered in execution engines."""
    print("[TEST] Checking face_openface3_embedding availability...\n")

    # Test JobProcessor
    job_processor = JobProcessor()
    assert "face_openface3_embedding" in job_processor.pipeline_classes, (
        "[ERROR] face_openface3_embedding NOT found in JobProcessor\n"
        f"Available: {sorted(job_processor.pipeline_classes.keys())}"
    )
    print("[OK] face_openface3_embedding found in JobProcessor (API)")

    # Test BatchOrchestrator
    storage = FileStorageBackend(Path("storage"))
    batch = BatchOrchestrator(storage_backend=storage)
    assert "face_openface3_embedding" in batch.pipeline_classes, (
        "[ERROR] face_openface3_embedding NOT found in BatchOrchestrator\n"
        f"Available: {sorted(batch.pipeline_classes.keys())}"
    )
    print("[OK] face_openface3_embedding found in BatchOrchestrator")

    # Test that it's the same class
    if (
        job_processor.pipeline_classes["face_openface3_embedding"]
        == batch.pipeline_classes["face_openface3_embedding"]
    ):
        print("[OK] Same pipeline class used in both engines")
    else:
        print(
            "[WARNING] Different pipeline classes in JobProcessor vs BatchOrchestrator"
        )

    print(f"\n[SUCCESS] face_openface3_embedding is properly registered and available!")
    return


if __name__ == "__main__":
    test_face_openface3_embedding()
    sys.exit(0)
