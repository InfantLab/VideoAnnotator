"""Person Identity Management System.

This module provides consistent person identification and labeling across all VideoAnnotator pipelines.
It ensures the same person receives the same ID across all visual analysis pipelines and supports
semantic labeling of person identities."""

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PersonLabel:
    """Person label information with confidence and metadata."""

    label: str
    confidence: float
    method: str = "manual"  # "manual", "automatic_size_based", "automatic_spatial"
    timestamp: float | None = None


class PersonIdentityManager:
    """
    Manages person identities across pipelines and videos using COCO standards.

    This class provides:
    - Consistent person ID assignment across video frames
    - Person labeling with confidence scores
    - Face-to-person linking using IoU matching
    - Cross-pipeline person identity sharing
    """

    def __init__(
        self,
        video_id: str = None,
        id_format: str = "semantic",
        config: dict[str, Any] = None,
    ):
        """Initialize person identity manager for a video.

        Args:
            video_id: Unique identifier for the video (can be None if using config)
            id_format: Either "semantic" (person_video_001) or "integer" (1, 2, 3)
            config: Optional configuration dictionary"""
        # Handle config parameter for compatibility with pipeline integration
        if config is not None and video_id is None:
            video_id = config.get("video_id", "default_video")
            id_format = config.get("id_format", id_format)

        self.video_id = video_id or "default_video"
        self.id_format = id_format
        self.track_to_person_map = {}  # track_id -> person_id
        self.person_labels = {}  # person_id -> PersonLabel
        self.next_person_id = 1
        self._person_tracks_cache = {}  # Cache for person track summaries

    def register_track(self, track_id: int) -> str:
        """
        Register a new track and assign person_id.

        Args:
            track_id: ByteTrack tracking ID

        Returns:
            Assigned person_id string
        """
        if track_id not in self.track_to_person_map:
            if self.id_format == "semantic":
                person_id = f"person_{self.video_id}_{self.next_person_id:03d}"
            else:  # integer format
                person_id = str(self.next_person_id)

            self.track_to_person_map[track_id] = person_id
            self.next_person_id += 1
            logger.debug(
                f"Registered new person: track_id={track_id} -> person_id={person_id}"
            )

        return self.track_to_person_map[track_id]

    def get_person_id(self, track_id: int) -> str | None:
        """Get person_id for a track_id."""
        return self.track_to_person_map.get(track_id)

    def set_person_label(
        self,
        person_id: str,
        label: str,
        confidence: float = 1.0,
        method: str = "manual",
        timestamp: float | None = None,
    ):
        """Assign semantic label to person.

        Args:
            person_id: Person identifier
            label: Semantic label (e.g., "infant", "parent")
            confidence: Confidence score (0.0 to 1.0)
            method: How the label was assigned
            timestamp: When label was assigned"""
        self.person_labels[person_id] = PersonLabel(
            label=label, confidence=confidence, method=method, timestamp=timestamp
        )
        logger.info(
            f"Set label for {person_id}: {label} (confidence={confidence:.2f}, method={method})"
        )

    def get_person_label(self, person_id: str) -> dict[str, Any] | None:
        """
        Get person label information.

        Returns:
            Dict with label, confidence, method, timestamp or None if not labeled
        """
        label_info = self.person_labels.get(person_id)
        if label_info:
            return {
                "label": label_info.label,
                "confidence": label_info.confidence,
                "method": label_info.method,
                "timestamp": label_info.timestamp,
            }
        return None

    def link_face_to_person(
        self,
        face_bbox: list[float],
        frame_annotations: list[dict],
        iou_threshold: float = 0.5,
    ) -> str | None:
        """
        Link face detection to person using IoU matching (COCO format).

        Args:
            face_bbox: Face bounding box [x, y, w, h]
            frame_annotations: List of person annotations for the frame
            iou_threshold: Minimum IoU for positive match

        Returns:
            person_id if match found, None otherwise
        """
        best_iou = 0.0
        best_person_id = None

        for annotation in frame_annotations:
            # Only match with person category annotations
            if annotation.get("category_id") == 1:  # Person category in COCO
                person_bbox = annotation.get("bbox", [])
                if len(person_bbox) == 4:
                    iou = self._calculate_iou(face_bbox, person_bbox)
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_person_id = annotation.get("person_id")

        if best_person_id:
            logger.debug(
                f"Linked face to person {best_person_id} with IoU={best_iou:.3f}"
            )

        return best_person_id

    def get_all_person_ids(self) -> list[str]:
        """Get list of all registered person IDs."""
        return list(self.track_to_person_map.values())

    def get_person_summary(self, person_id: str) -> dict[str, Any]:
        """
        Get summary information for a person.

        Returns:
            Dict with person metadata, label info, track summary
        """
        # Find corresponding track_id
        track_id = None
        for tid, pid in self.track_to_person_map.items():
            if pid == person_id:
                track_id = tid
                break

        summary = {
            "person_id": person_id,
            "track_id": track_id,
            "video_id": self.video_id,
        }

        # Add label information if available
        label_info = self.get_person_label(person_id)
        if label_info:
            summary.update(label_info)

        return summary

    def _calculate_iou(self, box1: list[float], box2: list[float]) -> float:
        """
        Calculate IoU between two bounding boxes [x, y, w, h].

        Args:
            box1: First bounding box [x, y, width, height]
            box2: Second bounding box [x, y, width, height]

        Returns:
            IoU score between 0.0 and 1.0
        """
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate intersection
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = (w1 * h1) + (w2 * h2) - intersection

        return intersection / union if union > 0 else 0.0

    @classmethod
    def from_person_tracks(
        cls, person_tracks: list[dict], video_id: str | None = None
    ) -> "PersonIdentityManager":
        """
        Create PersonIdentityManager from existing person tracking results.

        Args:
            person_tracks: List of person tracking annotations
            video_id: Video identifier (extracted from tracks if not provided)

        Returns:
            Initialized PersonIdentityManager
        """
        # Extract video_id from tracks if not provided
        if not video_id and person_tracks:
            # Try to extract from image_id pattern
            sample_annotation = person_tracks[0]
            image_id = sample_annotation.get("image_id", "")
            if "_frame_" in image_id:
                video_id = image_id.split("_frame_")[0]
            else:
                video_id = "unknown"

        manager = cls(video_id or "unknown")

        # Rebuild track-to-person mapping from existing data
        for annotation in person_tracks:
            track_id = annotation.get("track_id")
            person_id = annotation.get("person_id")

            if track_id is not None and person_id:
                manager.track_to_person_map[track_id] = person_id

                # Extract person labels if present
                person_label = annotation.get("person_label")
                label_confidence = annotation.get("label_confidence", 1.0)
                labeling_method = annotation.get("labeling_method", "unknown")

                if person_label and person_id not in manager.person_labels:
                    manager.set_person_label(
                        person_id, person_label, label_confidence, labeling_method
                    )

        # Update next_person_id counter
        if manager.track_to_person_map:
            existing_ids = list(manager.track_to_person_map.values())
            if manager.id_format == "semantic":
                # Extract numeric parts from semantic IDs
                max_id = 0
                for pid in existing_ids:
                    if "_" in pid:
                        try:
                            num_part = int(pid.split("_")[-1])
                            max_id = max(max_id, num_part)
                        except ValueError:
                            continue
                manager.next_person_id = max_id + 1
            else:
                # Integer format
                try:
                    max_id = max(int(pid) for pid in existing_ids if pid.isdigit())
                    manager.next_person_id = max_id + 1
                except ValueError:
                    manager.next_person_id = 1

        logger.info(
            f"Created PersonIdentityManager from {len(person_tracks)} tracks, "
            f"found {len(manager.track_to_person_map)} persons"
        )

        return manager

    def save_person_tracks(
        self, output_path: str, detections_summary: dict | None = None
    ):
        """Save person tracks and metadata to JSON file.

        Args:
            output_path: Output file path
            detections_summary: Optional summary of detection statistics"""
        person_tracks_data = []

        for track_id, person_id in self.track_to_person_map.items():
            track_data = {
                "person_id": person_id,
                "track_id": track_id,
                "video_id": self.video_id,
            }

            # Add label information if available
            label_info = self.get_person_label(person_id)
            if label_info:
                track_data.update(label_info)

            person_tracks_data.append(track_data)

        output_data = {
            "video_id": self.video_id,
            "person_tracks": person_tracks_data,
            "labeling_metadata": {
                "total_persons": len(self.track_to_person_map),
                "labeled_persons": len(self.person_labels),
                "id_format": self.id_format,
            },
        }

        if detections_summary:
            output_data["labeling_metadata"]["detections_summary"] = detections_summary

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved person tracks to {output_path}")

    @classmethod
    def load_person_tracks(cls, tracks_file: str) -> "PersonIdentityManager":
        """
        Load PersonIdentityManager from saved person tracks file.

        Args:
            tracks_file: Path to person tracks JSON file

        Returns:
            Loaded PersonIdentityManager
        """
        with open(tracks_file) as f:
            data = json.load(f)

        video_id = data.get("video_id", "unknown")
        metadata = data.get("labeling_metadata", {})
        id_format = metadata.get("id_format", "semantic")

        manager = cls(video_id, id_format)

        # Rebuild from person tracks
        person_tracks = data.get("person_tracks", [])
        for track_data in person_tracks:
            track_id = track_data.get("track_id")
            person_id = track_data.get("person_id")

            if track_id is not None and person_id:
                manager.track_to_person_map[track_id] = person_id

                # Restore labels
                if "label" in track_data:
                    manager.set_person_label(
                        person_id,
                        track_data["label"],
                        track_data.get("confidence", 1.0),
                        track_data.get("method", "manual"),
                        track_data.get("timestamp"),
                    )

        logger.info(
            f"Loaded PersonIdentityManager from {tracks_file}: "
            f"{len(manager.track_to_person_map)} persons"
        )

        return manager


# Person labeling schema and utilities
PERSON_LABELS = {
    # Family context
    "parent": {"aliases": ["mother", "father", "mom", "dad", "caregiver"]},
    "infant": {"aliases": ["baby", "child", "toddler", "kid"]},
    "sibling": {"aliases": ["brother", "sister"]},
    # Educational context
    "teacher": {"aliases": ["instructor", "educator", "professor"]},
    "student": {"aliases": ["pupil", "learner"]},
    # Clinical context
    "patient": {"aliases": ["client", "participant"]},
    "clinician": {"aliases": ["therapist", "doctor", "practitioner"]},
    # Generic
    "person": {"aliases": ["individual", "unknown", "adult"]},
}


def normalize_person_label(label: str) -> str | None:
    """
    Normalize person label to canonical form.

    Args:
        label: Input label (possibly an alias)

    Returns:
        Canonical label or None if not recognized
    """
    label_lower = label.lower().strip()

    # Check direct matches first
    if label_lower in PERSON_LABELS:
        return label_lower

    # Check aliases
    for canonical_label, info in PERSON_LABELS.items():
        if label_lower in info.get("aliases", []):
            return canonical_label

    return None


def get_available_labels() -> list[str]:
    """Get list of all available person labels."""
    return list(PERSON_LABELS.keys())


def get_label_aliases(label: str) -> list[str]:
    """Get list of aliases for a person label."""
    return PERSON_LABELS.get(label, {}).get("aliases", [])
