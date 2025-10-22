"""Single job processor for VideoAnnotator API.

Extracts core pipeline processing logic from BatchOrchestrator for API-
based individual job processing.
"""

from datetime import datetime

from .batch.types import BatchJob, JobStatus, PipelineResult
from .utils.logging_config import get_logger

logger = get_logger("api")


class JobProcessor:
    """Processes individual jobs through VideoAnnotator pipelines.

    This is a simplified version of BatchOrchestrator focused on single
    job processing for API use cases.
    """

    def __init__(self):
        """Initialize the job processor."""
        self.pipeline_classes = {}
        self._import_pipeline_classes()

    def _import_pipeline_classes(self):
        """Import all available pipeline classes."""
        try:
            # Import standard pipelines
            from pipelines.audio_processing import AudioPipeline
            from pipelines.audio_processing.laion_voice_pipeline import (
                LAIONVoicePipeline,
            )
            from pipelines.face_analysis import FaceAnalysisPipeline

            # Import LAION pipelines
            from pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            from pipelines.person_tracking import PersonTrackingPipeline
            from pipelines.scene_detection import SceneDetectionPipeline

            self.pipeline_classes = {
                "scene_detection": SceneDetectionPipeline,
                "scene": SceneDetectionPipeline,
                "person_tracking": PersonTrackingPipeline,
                "person": PersonTrackingPipeline,
                "face_analysis": FaceAnalysisPipeline,
                "face": FaceAnalysisPipeline,
                "audio_processing": AudioPipeline,
                "audio": AudioPipeline,
                "laion_face_analysis": LAIONFacePipeline,
                "laion_face": LAIONFacePipeline,
                "laion_voice_analysis": LAIONVoicePipeline,
                "laion_voice": LAIONVoicePipeline,
            }
            logger.debug("Pipeline classes imported successfully")

        except ImportError as e:
            logger.error(f"Failed to import pipeline classes: {e}")
            self.pipeline_classes = {}

    def process_job(self, job: BatchJob) -> bool:
        """Process a single job through selected pipelines.

        Args:
            job: Job to process

        Returns:
            True if successful, False if failed
        """
        try:
            logger.info(f"Processing job {job.job_id}: {job.video_path}")

            # Ensure video file exists
            if not job.video_path.exists():
                error_msg = f"Video file not found: {job.video_path}"
                logger.error(error_msg)
                job.error_message = error_msg
                return False

            # Generate output directory if not set
            if job.output_dir is None:
                # Create output directory next to the video file
                job.output_dir = job.video_path.parent / f"{job.video_path.stem}_output"
                logger.info(f"Generated output directory: {job.output_dir}")

            # Ensure output directory exists
            job.output_dir.mkdir(parents=True, exist_ok=True)

            # Determine pipelines to run
            pipelines_to_run = job.selected_pipelines or ["scene", "person"]
            logger.info(f"Running pipelines for job {job.job_id}: {pipelines_to_run}")

            # Process each pipeline
            for pipeline_name in pipelines_to_run:
                success = self._process_pipeline(job, pipeline_name)
                if not success:
                    return False

            logger.info(f"Job {job.job_id} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Exception during job processing: {e}", exc_info=True)
            job.error_message = str(e)
            return False

    def _process_pipeline(self, job: BatchJob, pipeline_name: str) -> bool:
        """Process a single pipeline for a job.

        Args:
            job: Job to process
            pipeline_name: Name of pipeline to run

        Returns:
            True if successful, False if failed
        """
        start_time = datetime.now()

        try:
            # Get pipeline class
            if pipeline_name not in self.pipeline_classes:
                error_msg = f"Unknown pipeline: {pipeline_name}"
                logger.error(error_msg)
                job.error_message = error_msg
                return False

            pipeline_class = self.pipeline_classes[pipeline_name]
            logger.info(f"Running {pipeline_name} pipeline for job {job.job_id}")

            # Initialize pipeline
            pipeline = pipeline_class()
            pipeline.initialize()

            try:
                # Handle different pipeline signatures
                if (
                    pipeline_name in ["audio_processing", "audio"]
                    and pipeline_class.__name__ == "AudioPipelineModular"
                ):
                    # AudioPipelineModular uses output_dir instead of pps
                    result = pipeline.process(
                        video_path=str(job.video_path),
                        start_time=0,
                        end_time=None,
                        output_dir=str(job.output_dir),
                    )
                else:
                    # Standard pipeline signature with pps
                    result = pipeline.process(
                        video_path=str(job.video_path),
                        start_time=0,
                        end_time=None,
                        pps=job.config.get("pps", 1) if job.config else 1,
                        output_dir=str(job.output_dir),
                    )

                end_time = datetime.now()

                # Create PipelineResult object
                pipeline_result = PipelineResult(
                    pipeline_name=pipeline_name,
                    status=JobStatus.COMPLETED,
                    start_time=start_time,
                    end_time=end_time,
                    annotation_count=len(result) if isinstance(result, list) else None,
                    output_file=None,  # Could be enhanced to track output files
                    error_message=None,
                )

                # Store result
                if not hasattr(job, "pipeline_results"):
                    job.pipeline_results = {}
                job.pipeline_results[pipeline_name] = pipeline_result

                logger.info(f"Pipeline {pipeline_name} completed for job {job.job_id}")
                return True

            finally:
                # Always cleanup pipeline resources
                try:
                    pipeline.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Pipeline cleanup error: {cleanup_error}")

        except Exception as e:
            end_time = datetime.now()

            # Create failed PipelineResult object
            pipeline_result = PipelineResult(
                pipeline_name=pipeline_name,
                status=JobStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
            )

            # Store failed result
            if not hasattr(job, "pipeline_results"):
                job.pipeline_results = {}
            job.pipeline_results[pipeline_name] = pipeline_result

            logger.error(
                f"Pipeline {pipeline_name} failed for job {job.job_id}: {e}",
                exc_info=True,
            )
            job.error_message = f"Pipeline {pipeline_name} failed: {e!s}"
            return False
