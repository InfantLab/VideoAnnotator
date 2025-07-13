"""
COCO format validation and utilities for VideoAnnotator.

This module provides pycocotools integration for validating and working with
COCO-format annotations from VideoAnnotator pipelines.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logging.warning("pycocotools not available. Install with: pip install pycocotools")

import numpy as np
from pydantic import BaseModel


class COCOValidationResult(BaseModel):
    """Result of COCO format validation."""
    is_valid: bool
    pipeline_name: str
    file_path: str
    num_images: int = 0
    num_annotations: int = 0
    num_categories: int = 0
    validation_errors: List[str] = []
    validation_warnings: List[str] = []


class COCOEvaluationResult(BaseModel):
    """Result of COCO evaluation metrics."""
    pipeline_name: str
    metric_type: str  # 'bbox', 'keypoints', 'segm'
    ap_50_95: float = 0.0  # Average Precision @ IoU=0.50:0.95
    ap_50: float = 0.0     # Average Precision @ IoU=0.50
    ap_75: float = 0.0     # Average Precision @ IoU=0.75
    ar_1: float = 0.0      # Average Recall with 1 detection per image
    ar_10: float = 0.0     # Average Recall with 10 detections per image
    ar_100: float = 0.0    # Average Recall with 100 detections per image
    stats: List[float] = []


class COCOValidator:
    """COCO format validator using pycocotools."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools is required. Install with: pip install pycocotools")
    
    def validate_coco_file(self, coco_json_path: str, pipeline_name: str = "unknown") -> COCOValidationResult:
        """
        Validate a COCO JSON file for format compliance.
        
        Args:
            coco_json_path: Path to COCO JSON file
            pipeline_name: Name of the pipeline that generated the file
            
        Returns:
            COCOValidationResult with validation status and details
        """
        result = COCOValidationResult(
            is_valid=False,
            pipeline_name=pipeline_name,
            file_path=coco_json_path
        )
        
        try:
            # Load JSON file
            with open(coco_json_path, 'r') as f:
                coco_data = json.load(f)
            
            # Basic structure validation
            errors = self._validate_coco_structure(coco_data)
            result.validation_errors.extend(errors)
            
            if errors:
                self.logger.error(f"COCO structure validation failed for {coco_json_path}: {errors}")
                return result
            
            # Initialize COCO object (this performs additional validation)
            coco = COCO(coco_json_path)
            
            # Extract statistics
            result.num_images = len(coco.getImgIds())
            result.num_annotations = len(coco.getAnnIds()) 
            result.num_categories = len(coco.getCatIds())
            
            # Additional VideoAnnotator-specific validation
            warnings = self._validate_video_extensions(coco_data)
            result.validation_warnings.extend(warnings)
            
            result.is_valid = True
            self.logger.info(f"COCO validation successful: {result.num_images} images, "
                           f"{result.num_annotations} annotations, {result.num_categories} categories")
            
        except Exception as e:
            error_msg = f"COCO validation failed: {str(e)}"
            result.validation_errors.append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def _validate_coco_structure(self, coco_data: Dict[str, Any]) -> List[str]:
        """Validate basic COCO JSON structure."""
        errors = []
        
        # Required top-level fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                errors.append(f"Missing required field: {field}")
        
        # Info field validation
        if 'info' in coco_data:
            info = coco_data['info']
            if not isinstance(info, dict):
                errors.append("'info' field must be a dictionary")
        
        # Images validation
        if 'images' in coco_data:
            for i, img in enumerate(coco_data['images']):
                if not isinstance(img, dict):
                    errors.append(f"Image {i} must be a dictionary")
                    continue
                    
                required_img_fields = ['id', 'width', 'height', 'file_name']
                for field in required_img_fields:
                    if field not in img:
                        errors.append(f"Image {i} missing required field: {field}")
        
        # Annotations validation
        if 'annotations' in coco_data:
            for i, ann in enumerate(coco_data['annotations']):
                if not isinstance(ann, dict):
                    errors.append(f"Annotation {i} must be a dictionary")
                    continue
                    
                required_ann_fields = ['id', 'image_id', 'category_id']
                for field in required_ann_fields:
                    if field not in ann:
                        errors.append(f"Annotation {i} missing required field: {field}")
                
                # Check bbox format if present
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        errors.append(f"Annotation {i} bbox must be [x, y, width, height]")
        
        # Categories validation
        if 'categories' in coco_data:
            for i, cat in enumerate(coco_data['categories']):
                if not isinstance(cat, dict):
                    errors.append(f"Category {i} must be a dictionary")
                    continue
                    
                required_cat_fields = ['id', 'name']
                for field in required_cat_fields:
                    if field not in cat:
                        errors.append(f"Category {i} missing required field: {field}")
        
        return errors
    
    def _validate_video_extensions(self, coco_data: Dict[str, Any]) -> List[str]:
        """Validate VideoAnnotator-specific COCO extensions."""
        warnings = []
        
        # Check for video-specific fields in images
        if 'images' in coco_data:
            for img in coco_data['images']:
                # VideoAnnotator often adds video_id, frame_number, timestamp
                video_fields = ['video_id', 'frame_number', 'timestamp']
                missing_video_fields = [f for f in video_fields if f not in img]
                if missing_video_fields:
                    warnings.append(f"Image {img.get('id', 'unknown')} missing video fields: {missing_video_fields}")
        
        # Check for tracking extensions in annotations
        if 'annotations' in coco_data:
            track_ids_found = any('track_id' in ann for ann in coco_data['annotations'])
            if not track_ids_found:
                warnings.append("No track_id fields found in annotations (expected for person tracking)")
        
        return warnings
    
    def evaluate_detection(self, gt_coco_path: str, pred_coco_path: str, 
                          pipeline_name: str = "detection") -> COCOEvaluationResult:
        """
        Evaluate object detection using COCO metrics.
        
        Args:
            gt_coco_path: Path to ground truth COCO JSON
            pred_coco_path: Path to predictions COCO JSON  
            pipeline_name: Name of the pipeline being evaluated
            
        Returns:
            COCOEvaluationResult with evaluation metrics
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools is required for evaluation")
        
        result = COCOEvaluationResult(
            pipeline_name=pipeline_name,
            metric_type="bbox"
        )
        
        try:
            # Load ground truth and predictions
            coco_gt = COCO(gt_coco_path)
            coco_pred = coco_gt.loadRes(pred_coco_path)
            
            # Initialize evaluator
            coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            stats = coco_eval.stats
            result.ap_50_95 = float(stats[0])  # AP @ IoU=0.50:0.95
            result.ap_50 = float(stats[1])     # AP @ IoU=0.50
            result.ap_75 = float(stats[2])     # AP @ IoU=0.75
            result.ar_1 = float(stats[6])      # AR @ IoU=0.50:0.95, max 1 det
            result.ar_10 = float(stats[7])     # AR @ IoU=0.50:0.95, max 10 det
            result.ar_100 = float(stats[8])    # AR @ IoU=0.50:0.95, max 100 det
            result.stats = [float(s) for s in stats]
            
            self.logger.info(f"Detection evaluation complete - AP@0.5:0.95: {result.ap_50_95:.3f}")
            
        except Exception as e:
            self.logger.error(f"Detection evaluation failed: {str(e)}")
            raise
        
        return result
    
    def evaluate_keypoints(self, gt_coco_path: str, pred_coco_path: str,
                          pipeline_name: str = "pose") -> COCOEvaluationResult:
        """
        Evaluate pose estimation using COCO keypoint metrics.
        
        Args:
            gt_coco_path: Path to ground truth COCO JSON
            pred_coco_path: Path to predictions COCO JSON
            pipeline_name: Name of the pipeline being evaluated
            
        Returns:
            COCOEvaluationResult with keypoint evaluation metrics
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools is required for evaluation")
        
        result = COCOEvaluationResult(
            pipeline_name=pipeline_name,
            metric_type="keypoints"
        )
        
        try:
            # Load ground truth and predictions
            coco_gt = COCO(gt_coco_path)
            coco_pred = coco_gt.loadRes(pred_coco_path)
            
            # Initialize keypoint evaluator
            coco_eval = COCOeval(coco_gt, coco_pred, 'keypoints')
            
            # Set COCO-17 keypoint OKS sigmas
            coco_eval.params.kpt_oks_sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                .62, 1.07, 1.07, .87, .87, .89, .89
            ]) / 10.0
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            stats = coco_eval.stats
            result.ap_50_95 = float(stats[0])
            result.ap_50 = float(stats[1])
            result.ap_75 = float(stats[2])
            result.ar_1 = float(stats[6])
            result.ar_10 = float(stats[7])
            result.ar_100 = float(stats[8])
            result.stats = [float(s) for s in stats]
            
            self.logger.info(f"Keypoint evaluation complete - AP@0.5:0.95: {result.ap_50_95:.3f}")
            
        except Exception as e:
            self.logger.error(f"Keypoint evaluation failed: {str(e)}")
            raise
        
        return result
    
    def validate_all_pipelines(self, annotations_dir: str) -> Dict[str, COCOValidationResult]:
        """
        Validate COCO files from all VideoAnnotator pipelines.
        
        Args:
            annotations_dir: Directory containing COCO JSON files
            
        Returns:
            Dictionary mapping pipeline names to validation results
        """
        results = {}
        annotations_path = Path(annotations_dir)
        
        # Expected COCO files from different pipelines
        pipeline_files = {
            'scene': '*scenes_coco.json',
            'person': '*person_detections_coco.json', 
            'face': '*face_detections_coco.json'
        }
        
        for pipeline_name, file_pattern in pipeline_files.items():
            coco_files = list(annotations_path.glob(file_pattern))
            
            if not coco_files:
                self.logger.warning(f"No COCO files found for {pipeline_name} pipeline with pattern {file_pattern}")
                results[pipeline_name] = COCOValidationResult(
                    is_valid=False,
                    pipeline_name=pipeline_name,
                    file_path="not_found",
                    validation_errors=[f"No files found matching pattern: {file_pattern}"]
                )
                continue
            
            # Validate the first file found (could be extended to validate all)
            coco_file = coco_files[0]
            results[pipeline_name] = self.validate_coco_file(str(coco_file), pipeline_name)
        
        return results
    
    def create_validation_report(self, results: Dict[str, COCOValidationResult]) -> str:
        """
        Create a human-readable validation report.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Formatted validation report as string
        """
        report_lines = [
            "🔍 VideoAnnotator COCO Validation Report",
            "=" * 50,
            ""
        ]
        
        total_pipelines = len(results)
        valid_pipelines = sum(1 for r in results.values() if r.is_valid)
        
        report_lines.extend([
            f"📊 Summary: {valid_pipelines}/{total_pipelines} pipelines valid",
            ""
        ])
        
        for pipeline_name, result in results.items():
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            report_lines.append(f"🔧 {pipeline_name.upper()} Pipeline: {status}")
            
            if result.is_valid:
                report_lines.extend([
                    f"   📁 File: {Path(result.file_path).name}",
                    f"   🖼️  Images: {result.num_images}",
                    f"   🏷️  Annotations: {result.num_annotations}",
                    f"   📂 Categories: {result.num_categories}"
                ])
                
                if result.validation_warnings:
                    report_lines.append("   ⚠️  Warnings:")
                    for warning in result.validation_warnings:
                        report_lines.append(f"     • {warning}")
            else:
                report_lines.append(f"   📁 File: {result.file_path}")
                if result.validation_errors:
                    report_lines.append("   ❌ Errors:")
                    for error in result.validation_errors:
                        report_lines.append(f"     • {error}")
            
            report_lines.append("")
        
        # Add recommendations
        if valid_pipelines < total_pipelines:
            report_lines.extend([
                "🛠️  Recommendations:",
                "• Check file paths and ensure COCO JSON files exist",
                "• Verify COCO format compliance using pycocotools.coco.COCO()",
                "• Review VideoAnnotator pipeline configuration",
                "• Consult PYCOCOTOOLS_INTEGRATION_PLAN.md for details",
                ""
            ])
        
        return "\n".join(report_lines)


# Convenience functions for common operations
def validate_coco_json(coco_json_path: str, pipeline_name: str = "unknown") -> COCOValidationResult:
    """Convenience function to validate a single COCO JSON file."""
    validator = COCOValidator()
    return validator.validate_coco_file(coco_json_path, pipeline_name)


def validate_pipeline_outputs(annotations_dir: str) -> Dict[str, COCOValidationResult]:
    """Convenience function to validate all pipeline COCO outputs."""
    validator = COCOValidator()
    return validator.validate_all_pipelines(annotations_dir)


def create_coco_evaluation_summary(gt_dir: str, pred_dir: str) -> Dict[str, COCOEvaluationResult]:
    """
    Create evaluation summary for all pipelines.
    
    Args:
        gt_dir: Directory containing ground truth COCO files
        pred_dir: Directory containing prediction COCO files
        
    Returns:
        Dictionary mapping pipeline names to evaluation results
    """
    validator = COCOValidator()
    results = {}
    
    # Detection evaluation (person pipeline)
    gt_person = Path(gt_dir) / "person_detections_gt_coco.json"
    pred_person = Path(pred_dir) / "person_detections_pred_coco.json"
    
    if gt_person.exists() and pred_person.exists():
        results['person_detection'] = validator.evaluate_detection(
            str(gt_person), str(pred_person), "person_detection"
        )
    
    # Keypoint evaluation (person pipeline with pose)
    gt_pose = Path(gt_dir) / "person_pose_gt_coco.json"
    pred_pose = Path(pred_dir) / "person_pose_pred_coco.json"
    
    if gt_pose.exists() and pred_pose.exists():
        results['person_pose'] = validator.evaluate_keypoints(
            str(gt_pose), str(pred_pose), "person_pose"
        )
    
    return results
