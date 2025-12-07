"""Verify consistency between pipeline registry and execution engines.

This script checks that all pipelines advertised in the registry metadata
have corresponding implementation classes in JobProcessor and BatchOrchestrator.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from videoannotator.api.job_processor import JobProcessor
from videoannotator.batch.batch_orchestrator import BatchOrchestrator
from videoannotator.registry.pipeline_registry import get_registry
from videoannotator.storage.file_backend import FileStorageBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Check pipeline consistency between registry and execution engines."""
    logger.info("[START] Verifying pipeline consistency...")

    # Load registry
    registry = get_registry()
    registry.load(force=True)
    registry_pipelines = {meta.name for meta in registry.list()}

    logger.info(f"[OK] Registry pipelines: {sorted(registry_pipelines)}")

    # Load JobProcessor pipelines
    job_processor = JobProcessor()
    job_processor_pipelines = set(job_processor.pipeline_classes.keys())

    logger.info(f"[OK] JobProcessor pipelines: {sorted(job_processor_pipelines)}")

    # Load BatchOrchestrator pipelines
    storage = FileStorageBackend(Path("storage"))
    batch_orchestrator = BatchOrchestrator(storage_backend=storage)
    batch_pipelines = set(batch_orchestrator.pipeline_classes.keys())

    logger.info(f"[OK] BatchOrchestrator pipelines: {sorted(batch_pipelines)}")

    # Check consistency - only check implemented pipelines
    # (pipelines in registry but not in execution engines)
    # missing_from_job_processor = registry_pipelines - job_processor_pipelines
    # missing_from_batch = registry_pipelines - batch_pipelines

    # Unimplemented pipelines (in registry but not available anywhere)
    unimplemented = registry_pipelines - (job_processor_pipelines | batch_pipelines)

    # Report results
    print("\n" + "=" * 70)
    print("PIPELINE CONSISTENCY REPORT")
    print("=" * 70)

    if unimplemented:
        print(
            f"[WARNING] Found {len(unimplemented)} unimplemented pipeline(s) in registry:"
        )
        for pipeline in sorted(unimplemented):
            print(f"    - {pipeline}")
        print("\n  These will be filtered out from API responses.")
        print("  To implement them, add pipeline classes and update importers.\n")

    # Check implemented pipelines for consistency
    implemented = registry_pipelines - unimplemented
    missing_from_job_processor_impl = implemented - job_processor_pipelines
    missing_from_batch_impl = implemented - batch_pipelines

    if not missing_from_job_processor_impl and not missing_from_batch_impl:
        print(
            f"[OK] All {len(implemented)} implemented registry pipelines are available!"
        )
        print(f"    - JobProcessor: {len(job_processor_pipelines)} pipelines")
        print(f"    - BatchOrchestrator: {len(batch_pipelines)} pipelines")
        return 0

    else:
        print("[ERROR] Found inconsistencies in implemented pipelines:")

        if missing_from_job_processor_impl:
            print("\n  Missing from JobProcessor (API):")
            for pipeline in sorted(missing_from_job_processor_impl):
                print(f"    - {pipeline}")

        if missing_from_batch_impl:
            print("\n  Missing from BatchOrchestrator:")
            for pipeline in sorted(missing_from_batch_impl):
                print(f"    - {pipeline}")

        print("\n[HINT] Add missing pipelines to _import_pipeline_classes() in:")
        print("       - src/videoannotator/api/job_processor.py")
        print("       - src/videoannotator/batch/batch_orchestrator.py")

        return 1


if __name__ == "__main__":
    sys.exit(main())
