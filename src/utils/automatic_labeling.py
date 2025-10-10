"""Automatic Person Labeling Utilities.

This module provides automatic labeling capabilities for person identification
based on visual cues like size, position, and spatial relationships."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AutomaticPersonLabeler:
    """
    Provides automatic person labeling based on visual and spatial cues.

    Labeling methods:
    - Size-based: Infer adult vs child based on bounding box dimensions
    - Position-based: Infer roles based on spatial positioning
    - Activity-based: Infer roles based on movement patterns
    """

    def __init__(self, config: dict | None = None):
        """Initialize automatic labeler with configuration.

        Args:
            config: Configuration dict with labeling parameters"""
        self.config = config or self._default_config()

    def _default_config(self) -> dict:
        """Default configuration for automatic labeling."""
        return {
            "size_based": {
                "enabled": True,
                "height_threshold": 0.4,  # Normalized height threshold for child/adult
                "confidence": 0.7,
                "adult_label": "parent",
                "child_label": "infant",
            },
            "position_based": {
                "enabled": True,
                "center_bias_threshold": 0.7,  # Threshold for center positioning
                "confidence": 0.6,
                "primary_label": "infant",  # Center subject is often the child
                "secondary_label": "parent",
            },
            "temporal_consistency": {
                "enabled": True,
                "min_detections": 10,  # Minimum detections for stable labeling
                "consistency_threshold": 0.8,
            },
        }

    def label_persons_automatic(
        self, person_tracks: list[dict], frame_annotations: list[dict]
    ) -> dict[str, dict]:
        """
        Automatically label persons based on visual cues.

        Args:
            person_tracks: List of person track summaries
            frame_annotations: All frame annotations for analysis

        Returns:
            Dict mapping person_id to label information
        """
        labels = {}

        # Group annotations by person_id
        person_detections = self._group_detections_by_person(frame_annotations)

        # Apply size-based labeling
        if self.config["size_based"]["enabled"]:
            size_labels = self._apply_size_based_labeling(person_detections)
            labels.update(size_labels)

        # Apply position-based labeling
        if self.config["position_based"]["enabled"]:
            position_labels = self._apply_position_based_labeling(person_detections)
            labels.update(position_labels)

        # Apply temporal consistency filtering
        if self.config["temporal_consistency"]["enabled"]:
            labels = self._apply_temporal_consistency(labels, person_detections)

        logger.info(f"Automatic labeling assigned {len(labels)} person labels")
        return labels

    def _group_detections_by_person(
        self, frame_annotations: list[dict]
    ) -> dict[str, list[dict]]:
        """Group frame annotations by person_id."""
        person_detections = {}

        for annotation in frame_annotations:
            person_id = annotation.get("person_id")
            if person_id:
                if person_id not in person_detections:
                    person_detections[person_id] = []
                person_detections[person_id].append(annotation)

        return person_detections

    def _apply_size_based_labeling(
        self, person_detections: dict[str, list[dict]]
    ) -> dict[str, dict]:
        """
        Apply size-based automatic labeling.

        Logic: Smaller bounding boxes typically indicate children/infants,
               larger bounding boxes indicate adults/parents.
        """
        labels = {}
        config = self.config["size_based"]

        # Calculate average heights for each person
        person_heights = {}
        for person_id, detections in person_detections.items():
            heights = []
            for detection in detections:
                bbox = detection.get("bbox", [])
                if len(bbox) >= 4:
                    height = bbox[3]  # Height is the 4th element [x, y, w, h]
                    heights.append(height)

            if heights:
                avg_height = np.mean(heights)
                person_heights[person_id] = avg_height

        # Normalize heights relative to maximum
        if person_heights:
            max_height = max(person_heights.values())
            if max_height > 0:
                for person_id, height in person_heights.items():
                    normalized_height = height / max_height

                    if normalized_height < config["height_threshold"]:
                        label = config["child_label"]
                    else:
                        label = config["adult_label"]

                    labels[person_id] = {
                        "label": label,
                        "confidence": config["confidence"],
                        "method": "automatic_size_based",
                        "metadata": {
                            "normalized_height": normalized_height,
                            "raw_height": height,
                            "height_threshold": config["height_threshold"],
                        },
                    }

                    logger.debug(
                        f"Size-based label: {person_id} -> {label} "
                        f"(height={normalized_height:.3f})"
                    )

        return labels

    def _apply_position_based_labeling(
        self, person_detections: dict[str, list[dict]]
    ) -> dict[str, dict]:
        """
        Apply position-based automatic labeling.

        Logic: Person consistently in center is often the primary subject (child),
               persons at edges are often secondary participants (adults/caregivers).
        """
        labels = {}
        config = self.config["position_based"]

        # Calculate center bias for each person
        person_center_bias = {}
        for person_id, detections in person_detections.items():
            center_scores = []

            for detection in detections:
                bbox = detection.get("bbox", [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox
                    # Calculate center of bounding box
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # Assume frame dimensions (could be extracted from image_id or metadata)
                    # For now, use normalized coordinates assumption
                    if center_x <= 1.0 and center_y <= 1.0:  # Already normalized
                        frame_center_x, frame_center_y = 0.5, 0.5
                    else:  # Absolute coordinates - would need frame dimensions
                        # Skip position-based labeling for absolute coordinates
                        continue

                    # Calculate distance from frame center
                    distance_from_center = np.sqrt(
                        (center_x - frame_center_x) ** 2
                        + (center_y - frame_center_y) ** 2
                    )

                    # Convert to center bias score (0 = edge, 1 = center)
                    max_distance = np.sqrt(0.5**2 + 0.5**2)  # Corner to center
                    center_score = 1.0 - (distance_from_center / max_distance)
                    center_scores.append(center_score)

            if center_scores:
                avg_center_bias = np.mean(center_scores)
                person_center_bias[person_id] = avg_center_bias

        # Apply position-based labeling
        for person_id, center_bias in person_center_bias.items():
            # Only apply if we don't already have a size-based label with higher confidence
            if (
                person_id not in labels
                or labels[person_id]["confidence"] < config["confidence"]
            ):
                if center_bias > config["center_bias_threshold"]:
                    label = config["primary_label"]
                else:
                    label = config["secondary_label"]

                labels[person_id] = {
                    "label": label,
                    "confidence": config["confidence"],
                    "method": "automatic_position_based",
                    "metadata": {
                        "center_bias": center_bias,
                        "center_bias_threshold": config["center_bias_threshold"],
                    },
                }

                logger.debug(
                    f"Position-based label: {person_id} -> {label} "
                    f"(center_bias={center_bias:.3f})"
                )

        return labels

    def _apply_temporal_consistency(
        self, labels: dict[str, dict], person_detections: dict[str, list[dict]]
    ) -> dict[str, dict]:
        """
        Apply temporal consistency filtering to automatic labels.

        Only keep labels for persons with sufficient detections.
        """
        config = self.config["temporal_consistency"]
        filtered_labels = {}

        for person_id, label_info in labels.items():
            detection_count = len(person_detections.get(person_id, []))

            if detection_count >= config["min_detections"]:
                # Adjust confidence based on detection count
                detection_confidence = min(
                    1.0, detection_count / (config["min_detections"] * 2)
                )
                adjusted_confidence = label_info["confidence"] * detection_confidence

                if adjusted_confidence >= config["consistency_threshold"]:
                    label_info = label_info.copy()
                    label_info["confidence"] = adjusted_confidence
                    label_info["metadata"]["detection_count"] = detection_count
                    label_info["metadata"]["temporal_consistency_applied"] = True

                    filtered_labels[person_id] = label_info

                    logger.debug(
                        f"Temporal consistency: {person_id} kept with "
                        f"confidence={adjusted_confidence:.3f} "
                        f"({detection_count} detections)"
                    )
                else:
                    logger.debug(
                        f"Temporal consistency: {person_id} filtered out "
                        f"(low confidence={adjusted_confidence:.3f})"
                    )
            else:
                logger.debug(
                    f"Temporal consistency: {person_id} filtered out "
                    f"(insufficient detections={detection_count})"
                )

        return filtered_labels

    def analyze_spatial_relationships(
        self, person_detections: dict[str, list[dict]]
    ) -> dict[str, Any]:
        """
        Analyze spatial relationships between persons for additional context.

        Returns:
            Dict with spatial relationship analysis
        """
        analysis = {
            "person_count": len(person_detections),
            "spatial_patterns": {},
            "interaction_likelihood": {},
        }

        if len(person_detections) < 2:
            return analysis

        # Analyze relative positioning patterns
        person_ids = list(person_detections.keys())

        for i, person_id_1 in enumerate(person_ids):
            for person_id_2 in person_ids[i + 1 :]:
                detections_1 = person_detections[person_id_1]
                detections_2 = person_detections[person_id_2]

                # Calculate average relative positions
                relative_positions = []
                for det1 in detections_1:
                    for det2 in detections_2:
                        # Find temporally close detections
                        t1 = det1.get("timestamp", det1.get("frame_number", 0))
                        t2 = det2.get("timestamp", det2.get("frame_number", 0))

                        if abs(t1 - t2) <= 1.0:  # Within 1 second/frame
                            bbox1 = det1.get("bbox", [])
                            bbox2 = det2.get("bbox", [])

                            if len(bbox1) >= 4 and len(bbox2) >= 4:
                                # Calculate center-to-center distance
                                center1 = [
                                    bbox1[0] + bbox1[2] / 2,
                                    bbox1[1] + bbox1[3] / 2,
                                ]
                                center2 = [
                                    bbox2[0] + bbox2[2] / 2,
                                    bbox2[1] + bbox2[3] / 2,
                                ]

                                distance = np.sqrt(
                                    (center1[0] - center2[0]) ** 2
                                    + (center1[1] - center2[1]) ** 2
                                )
                                relative_positions.append(distance)

                if relative_positions:
                    avg_distance = np.mean(relative_positions)
                    pair_key = f"{person_id_1}_{person_id_2}"

                    analysis["spatial_patterns"][pair_key] = {
                        "average_distance": avg_distance,
                        "measurement_count": len(relative_positions),
                        "interaction_likelihood": "high"
                        if avg_distance < 0.3
                        else "low",
                    }

        return analysis


def infer_person_labels_from_tracks(
    person_tracks: list[dict], frame_annotations: list[dict], config: dict | None = None
) -> dict[str, dict]:
    """
    Convenience function to infer person labels from tracking data.

    Args:
        person_tracks: List of person track summaries
        frame_annotations: All frame annotations for analysis
        config: Optional configuration for labeling

    Returns:
        Dict mapping person_id to label information
    """
    labeler = AutomaticPersonLabeler(config)
    return labeler.label_persons_automatic(person_tracks, frame_annotations)


def calculate_labeling_confidence(
    person_detections: dict[str, list[dict]],
) -> dict[str, float]:
    """
    Calculate overall confidence scores for automatic labeling.

    Args:
        person_detections: Grouped detections by person_id

    Returns:
        Dict mapping person_id to confidence score
    """
    confidence_scores = {}

    for person_id, detections in person_detections.items():
        detection_count = len(detections)

        # Base confidence on detection count
        if detection_count >= 50:
            confidence = 0.9
        elif detection_count >= 20:
            confidence = 0.8
        elif detection_count >= 10:
            confidence = 0.7
        elif detection_count >= 5:
            confidence = 0.6
        else:
            confidence = 0.5

        # Adjust for temporal consistency
        if detections:
            timestamps = [
                d.get("timestamp", d.get("frame_number", 0)) for d in detections
            ]
            time_span = max(timestamps) - min(timestamps)

            if time_span > 30:  # Good temporal coverage
                confidence *= 1.1
            elif time_span < 5:  # Limited temporal coverage
                confidence *= 0.9

        confidence_scores[person_id] = min(1.0, confidence)

    return confidence_scores
