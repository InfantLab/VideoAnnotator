#!/usr/bin/env python3
"""
Person Labeling Tool

Command-line utility for reviewing and labeling persons in videos
using the VideoAnnotator person tracking system.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.utils.person_identity import (
    PersonIdentityManager,
    get_available_labels,
    normalize_person_label,
)
from src.version import __version__


def load_person_tracking_results(
    video_path: str, results_dir: str
) -> list[dict] | None:
    """Load existing person tracking results."""
    video_id = Path(video_path).stem
    tracking_file = Path(results_dir) / f"{video_id}_person_tracking.json"

    if not tracking_file.exists():
        print(f"No person tracking results found at: {tracking_file}")
        print("Please run person tracking first.")
        return None

    with open(tracking_file) as f:
        data = json.load(f)

    return data.get("annotations", [])


def load_existing_person_tracks(
    video_path: str, results_dir: str
) -> PersonIdentityManager | None:
    """Load existing person tracks if available."""
    video_id = Path(video_path).stem
    tracks_file = Path(results_dir) / f"{video_id}_person_tracks.json"

    if tracks_file.exists():
        try:
            return PersonIdentityManager.load_person_tracks(str(tracks_file))
        except Exception as e:
            print(f"Warning: Could not load existing person tracks: {e}")

    return None


def analyze_persons_in_video(annotations: list[dict]) -> dict[str, dict]:
    """Analyze persons found in video tracking results."""
    person_stats = {}

    for annotation in annotations:
        person_id = annotation.get("person_id")
        if person_id:
            if person_id not in person_stats:
                person_stats[person_id] = {
                    "detections": 0,
                    "first_frame": float("inf"),
                    "last_frame": 0,
                    "avg_bbox_area": 0,
                    "bbox_areas": [],
                    "current_label": None,
                    "label_confidence": None,
                }

            stats = person_stats[person_id]
            stats["detections"] += 1

            frame_num = annotation.get("frame_number", 0)
            stats["first_frame"] = min(stats["first_frame"], frame_num)
            stats["last_frame"] = max(stats["last_frame"], frame_num)

            bbox = annotation.get("bbox", [])
            if len(bbox) >= 4:
                area = bbox[2] * bbox[3]  # width * height
                stats["bbox_areas"].append(area)

            # Check for existing labels
            if annotation.get("person_label"):
                stats["current_label"] = annotation.get("person_label")
                stats["label_confidence"] = annotation.get("label_confidence", 1.0)

    # Calculate average areas
    for _person_id, stats in person_stats.items():
        if stats["bbox_areas"]:
            stats["avg_bbox_area"] = sum(stats["bbox_areas"]) / len(stats["bbox_areas"])

    return person_stats


def display_person_analysis(person_stats: dict[str, dict]):
    """Display analysis of persons in the video."""
    print("\n" + "=" * 70)
    print("PERSON ANALYSIS")
    print("=" * 70)

    # Sort by detection count (most prominent first)
    sorted_persons = sorted(
        person_stats.items(), key=lambda x: x[1]["detections"], reverse=True
    )

    for i, (person_id, stats) in enumerate(sorted_persons, 1):
        print(f"\n{i}. Person ID: {person_id}")
        print(f"   Detections: {stats['detections']}")
        print(f"   Frame range: {stats['first_frame']} - {stats['last_frame']}")
        print(f"   Avg bbox area: {stats['avg_bbox_area']:.1f}")

        if stats["current_label"]:
            print(
                f"   Current label: {stats['current_label']} "
                f"(confidence: {stats['label_confidence']:.2f})"
            )
        else:
            print("   Current label: <not labeled>")


def interactive_labeling_session(
    identity_manager: PersonIdentityManager, person_stats: dict[str, dict]
):
    """Run interactive labeling session."""
    print("\n" + "=" * 70)
    print("INTERACTIVE LABELING SESSION")
    print("=" * 70)
    print("Available labels:", ", ".join(get_available_labels()))
    print("Type 'help' for commands, 'quit' to finish")

    while True:
        print("\n" + "-" * 50)

        # Show unlabeled persons
        unlabeled = [
            pid for pid, stats in person_stats.items() if not stats.get("current_label")
        ]

        if unlabeled:
            print(f"Unlabeled persons: {', '.join(unlabeled)}")
        else:
            print("All persons have been labeled!")

        command = input(
            "\nEnter command (person_id label, 'list', 'help', 'quit'): "
        ).strip()

        if command.lower() in ["quit", "q", "exit"]:
            break
        elif command.lower() in ["help", "h"]:
            print_help()
        elif command.lower() in ["list", "l"]:
            display_person_analysis(person_stats)
        elif " " in command:
            # Parse labeling command
            parts = command.split()
            person_id = parts[0]
            label = parts[1].lower()

            if person_id not in person_stats:
                print(f"Error: Person ID '{person_id}' not found")
                continue

            normalized_label = normalize_person_label(label)
            if not normalized_label:
                print(f"Error: Label '{label}' not recognized")
                print(f"Available labels: {', '.join(get_available_labels())}")
                continue

            # Apply label
            identity_manager.set_person_label(
                person_id, normalized_label, 1.0, "manual"
            )
            person_stats[person_id]["current_label"] = normalized_label
            person_stats[person_id]["label_confidence"] = 1.0

            print(f"✓ Labeled {person_id} as '{normalized_label}'")
        else:
            print("Invalid command. Type 'help' for usage information.")


def print_help():
    """Print help information."""
    print("\nCOMMANDS:")
    print("  person_id label  - Label a person (e.g., 'person_video_001 infant')")
    print("  list             - Show person analysis")
    print("  help             - Show this help")
    print("  quit             - Finish labeling session")
    print(f"\nAVAILABLE LABELS: {', '.join(get_available_labels())}")


def apply_bulk_labeling(
    identity_manager: PersonIdentityManager,
    person_stats: dict[str, dict],
    bulk_rules: str,
):
    """Apply bulk labeling rules."""
    print(f"\nApplying bulk labeling rules: {bulk_rules}")

    # Parse bulk rules (e.g., "largest=parent,smallest=infant")
    rules = bulk_rules.split(",")

    for rule in rules:
        if "=" not in rule:
            continue

        criterion, label = rule.split("=", 1)
        criterion = criterion.strip().lower()
        label = label.strip().lower()

        normalized_label = normalize_person_label(label)
        if not normalized_label:
            print(f"Warning: Label '{label}' not recognized, skipping rule")
            continue

        # Apply different bulk labeling criteria
        if criterion == "largest":
            # Label person with largest average bbox area
            largest_person = max(
                person_stats.items(), key=lambda x: x[1]["avg_bbox_area"]
            )
            person_id = largest_person[0]
            identity_manager.set_person_label(
                person_id, normalized_label, 0.8, "bulk_largest"
            )
            person_stats[person_id]["current_label"] = normalized_label
            person_stats[person_id]["label_confidence"] = 0.8
            print(f"✓ Labeled largest person {person_id} as '{normalized_label}'")

        elif criterion == "smallest":
            # Label person with smallest average bbox area
            smallest_person = min(
                person_stats.items(), key=lambda x: x[1]["avg_bbox_area"]
            )
            person_id = smallest_person[0]
            identity_manager.set_person_label(
                person_id, normalized_label, 0.8, "bulk_smallest"
            )
            person_stats[person_id]["current_label"] = normalized_label
            person_stats[person_id]["label_confidence"] = 0.8
            print(f"✓ Labeled smallest person {person_id} as '{normalized_label}'")

        elif criterion == "most_detections":
            # Label person with most detections
            most_detected = max(person_stats.items(), key=lambda x: x[1]["detections"])
            person_id = most_detected[0]
            identity_manager.set_person_label(
                person_id, normalized_label, 0.8, "bulk_most_detections"
            )
            person_stats[person_id]["current_label"] = normalized_label
            person_stats[person_id]["label_confidence"] = 0.8
            print(f"✓ Labeled most detected person {person_id} as '{normalized_label}'")


def save_updated_annotations(
    annotations: list[dict], identity_manager: PersonIdentityManager, output_file: str
):
    """Save annotations with updated person labels."""
    # Update annotations with new labels
    for annotation in annotations:
        person_id = annotation.get("person_id")
        if person_id:
            label_info = identity_manager.get_person_label(person_id)
            if label_info:
                annotation["person_label"] = label_info["label"]
                annotation["label_confidence"] = label_info["confidence"]
                annotation["labeling_method"] = label_info["method"]

    # Save updated tracking results
    output_data = {
        "annotations": annotations,
        "info": {
            "description": "Person tracking results with manual labels",
            "version": __version__,
            "date_labeled": str(Path().cwd()),
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved updated annotations to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Person labeling tool for VideoAnnotator"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--results-dir",
        "-r",
        default="./results",
        help="Directory containing tracking results",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./labeled_results",
        help="Output directory for labeled results",
    )
    parser.add_argument(
        "--bulk-label",
        "-b",
        help="Bulk labeling rules (e.g., 'largest=parent,smallest=infant')",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Only run automatic labeling, skip interactive session",
    )
    parser.add_argument(
        "--run-tracking",
        action="store_true",
        help="Run person tracking if results don't exist",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing video: {video_path}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Load or run person tracking
    annotations = load_person_tracking_results(str(video_path), str(results_dir))

    if annotations is None:
        if args.run_tracking:
            print("\nRunning person tracking...")
            pipeline = PersonTrackingPipeline()
            annotations = pipeline.process(str(video_path), output_dir=str(results_dir))
            print(f"Person tracking complete: {len(annotations)} detections")
        else:
            print("Use --run-tracking to automatically run person tracking")
            return 1

    # Load or create identity manager
    identity_manager = load_existing_person_tracks(str(video_path), str(results_dir))
    if identity_manager is None:
        video_id = video_path.stem
        identity_manager = PersonIdentityManager.from_person_tracks(
            annotations, video_id
        )
        print(f"Created new PersonIdentityManager for {video_id}")
    else:
        print("Loaded existing person tracks")

    # Analyze persons in video
    person_stats = analyze_persons_in_video(annotations)
    if not person_stats:
        print("No persons found in tracking results")
        return 1

    display_person_analysis(person_stats)

    # Apply bulk labeling if specified
    if args.bulk_label:
        apply_bulk_labeling(identity_manager, person_stats, args.bulk_label)
        display_person_analysis(person_stats)

    # Run interactive labeling session
    if not args.auto_only:
        interactive_labeling_session(identity_manager, person_stats)

    # Save updated results
    video_id = video_path.stem

    # Save person tracks
    person_tracks_file = output_dir / f"{video_id}_person_tracks.json"
    identity_manager.save_person_tracks(str(person_tracks_file))

    # Save updated annotations
    updated_annotations_file = output_dir / f"{video_id}_person_tracking_labeled.json"
    save_updated_annotations(
        annotations, identity_manager, str(updated_annotations_file)
    )

    print("\n✓ Person labeling complete!")
    print(f"  Person tracks: {person_tracks_file}")
    print(f"  Labeled annotations: {updated_annotations_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
