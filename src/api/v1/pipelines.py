"""
Pipeline information endpoints for VideoAnnotator API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

# TODO: Import config system after fixing dependencies
# from ...config import load_config


router = APIRouter()


class PipelineInfo(BaseModel):
    """Information about an available pipeline."""
    name: str
    description: str
    enabled: bool
    config_schema: Dict[str, Any]


class PipelineListResponse(BaseModel):
    """Response for pipeline listing."""
    pipelines: List[PipelineInfo]
    total: int


@router.get("/", response_model=PipelineListResponse)
async def list_pipelines():
    """
    List all available pipelines.
    
    Returns:
        List of available pipelines with their configurations
    """
    try:
        # TODO: Load actual configuration when dependencies are fixed
        # For now, return mock pipeline information
        pipelines = []
        
        # Mock pipeline information (TODO: Load from actual config)
        pipelines.append(PipelineInfo(
            name="scene_detection",
            description="Detect scene boundaries and classify environments using PySceneDetect + CLIP",
            enabled=True,
            config_schema={
                "threshold": {"type": "float", "default": 30.0, "description": "Scene change threshold"},
                "min_scene_length": {"type": "float", "default": 1.0, "description": "Minimum scene length in seconds"},
                "enabled": {"type": "boolean", "default": True, "description": "Enable/disable pipeline"}
            }
        ))
        
        pipelines.append(PipelineInfo(
            name="person_tracking",
            description="Track people across frames with YOLO11 + ByteTrack pose estimation",
            enabled=True,
            config_schema={
                "model": {"type": "string", "default": "yolo11n-pose.pt", "description": "YOLO model to use"},
                "conf_threshold": {"type": "float", "default": 0.4, "description": "Confidence threshold"},
                "iou_threshold": {"type": "float", "default": 0.7, "description": "IoU threshold"},
                "track_mode": {"type": "boolean", "default": True, "description": "Enable tracking"}
            }
        ))
        
        pipelines.append(PipelineInfo(
            name="face_analysis",
            description="Multi-backend face analysis with OpenFace 3.0, LAION Face, and emotion detection",
            enabled=True,
            config_schema={
                "backend": {"type": "string", "default": "openface", "description": "Face analysis backend"},
                "confidence_threshold": {"type": "float", "default": 0.5, "description": "Face detection confidence"},
                "enable_emotions": {"type": "boolean", "default": True, "description": "Enable emotion analysis"}
            }
        ))
        
        pipelines.append(PipelineInfo(
            name="audio_processing",
            description="Speech recognition and speaker diarization with Whisper + pyannote.audio",
            enabled=True,
            config_schema={
                "whisper_model": {"type": "string", "default": "base", "description": "Whisper model size"},
                "enable_diarization": {"type": "boolean", "default": True, "description": "Enable speaker diarization"},
                "min_speakers": {"type": "integer", "default": 1, "description": "Minimum number of speakers"},
                "max_speakers": {"type": "integer", "default": 10, "description": "Maximum number of speakers"}
            }
        ))
        
        return PipelineListResponse(
            pipelines=pipelines,
            total=len(pipelines)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list pipelines: {str(e)}"
        )


@router.get("/{pipeline_name}", response_model=PipelineInfo)
async def get_pipeline_info(pipeline_name: str):
    """
    Get detailed information about a specific pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Detailed pipeline information
    """
    try:
        # Get all pipelines
        pipelines_response = await list_pipelines()
        
        # Find the requested pipeline
        for pipeline in pipelines_response.pipelines:
            if pipeline.name == pipeline_name:
                return pipeline
        
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_name}' not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pipeline info: {str(e)}"
        )


@router.post("/{pipeline_name}/validate")
async def validate_pipeline_config(pipeline_name: str, config: Dict[str, Any]):
    """
    Validate a pipeline configuration.
    
    Args:
        pipeline_name: Name of the pipeline
        config: Configuration to validate
        
    Returns:
        Validation result
    """
    try:
        # Get pipeline info to check if it exists
        pipeline_info = await get_pipeline_info(pipeline_name)
        
        # TODO: Implement proper config validation against schema
        # For now, just check if the pipeline exists and return success
        
        return {
            "valid": True,
            "pipeline": pipeline_name,
            "message": "Configuration is valid"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate config: {str(e)}"
        )