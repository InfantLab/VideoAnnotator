#!/usr/bin/env python3
"""
Size-Based Person Analysis Demo

Demonstrates the first version of automated person analysis using size-based inference.
This script shows how to use the simplified size-based analyzer to classify persons as adults or children.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.size_based_person_analysis import (
    SizeBasedPersonAnalyzer,
    demo_size_based_analysis,
    print_analysis_results,
    run_size_based_analysis,
)


def create_sample_data():
    """Create sample person tracking data for demonstration."""

    print("Creating sample person tracking data...")

    # Simulate person tracking results from a parent-child interaction video
    sample_annotations = [
        # Adult person (person_001) - consistently larger bounding boxes
        {
            "person_id": "person_babyvideo_001",
            "bbox": [150, 100, 100, 200],
            "frame_number": 1,
            "timestamp": 0.5,
        },
        {
            "person_id": "person_babyvideo_001",
            "bbox": [155, 105, 95, 195],
            "frame_number": 5,
            "timestamp": 2.5,
        },
        {
            "person_id": "person_babyvideo_001",
            "bbox": [160, 110, 98, 200],
            "frame_number": 10,
            "timestamp": 5.0,
        },
        {
            "person_id": "person_babyvideo_001",
            "bbox": [145, 95, 102, 205],
            "frame_number": 15,
            "timestamp": 7.5,
        },
        # Child person (person_002) - consistently smaller bounding boxes
        {
            "person_id": "person_babyvideo_002",
            "bbox": [300, 180, 60, 90],
            "frame_number": 1,
            "timestamp": 0.5,
        },
        {
            "person_id": "person_babyvideo_002",
            "bbox": [305, 175, 58, 88],
            "frame_number": 5,
            "timestamp": 2.5,
        },
        {
            "person_id": "person_babyvideo_002",
            "bbox": [298, 185, 62, 92],
            "frame_number": 10,
            "timestamp": 5.0,
        },
        {
            "person_id": "person_babyvideo_002",
            "bbox": [302, 178, 59, 89],
            "frame_number": 15,
            "timestamp": 7.5,
        },
        # Another adult (person_003) - similar size to person_001
        {
            "person_id": "person_babyvideo_003",
            "bbox": [50, 120, 105, 190],
            "frame_number": 8,
            "timestamp": 4.0,
        },
        {
            "person_id": "person_babyvideo_003",
            "bbox": [55, 115, 100, 195],
            "frame_number": 12,
            "timestamp": 6.0,
        },
        {
            "person_id": "person_babyvideo_003",
            "bbox": [48, 125, 108, 188],
            "frame_number": 16,
            "timestamp": 8.0,
        },
    ]

    print(f"Created {len(sample_annotations)} detection annotations for 3 persons")
    return sample_annotations


def demo_basic_analysis():
    """Demonstrate basic size-based analysis."""

    print("\n" + "=" * 60)
    print("DEMO 1: Basic Size-Based Analysis")
    print("=" * 60)

    # Create sample data
    annotations = create_sample_data()

    # Run analysis with default settings
    results = run_size_based_analysis(annotations)

    # Print detailed results
    print_analysis_results(results)

    return results


def demo_custom_threshold():
    """Demonstrate analysis with custom threshold."""

    print("\n" + "=" * 60)
    print("DEMO 2: Custom Threshold Analysis")
    print("=" * 60)

    # Create sample data
    annotations = create_sample_data()

    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6]

    for threshold in thresholds:
        print(f"\n--- Testing with height_threshold = {threshold} ---")

        results = run_size_based_analysis(
            annotations, height_threshold=threshold, confidence=0.8
        )

        # Print summary
        infant_count = sum(1 for info in results.values() if info["label"] == "infant")
        parent_count = sum(1 for info in results.values() if info["label"] == "parent")

        print(f"Results: {infant_count} infants, {parent_count} parents")

        for person_id, info in results.items():
            height = info["metadata"]["normalized_height"]
            print(f"  {person_id}: {info['label']} (height={height:.2f})")


def demo_analyzer_class():
    """Demonstrate using the analyzer class directly."""

    print("\n" + "=" * 60)
    print("DEMO 3: Using SizeBasedPersonAnalyzer Class")
    print("=" * 60)

    # Create analyzer with custom settings
    analyzer = SizeBasedPersonAnalyzer(height_threshold=0.45, confidence=0.85)

    # Create sample data
    annotations = create_sample_data()

    print(
        f"Analyzer settings: threshold={analyzer.height_threshold}, confidence={analyzer.confidence}"
    )

    # Run analysis
    results = analyzer.analyze_persons(annotations)

    # Print results
    print_analysis_results(results)

    # Show detailed metadata for one person
    if results:
        sample_person = next(iter(results.keys()))
        sample_info = results[sample_person]

        print(f"\nDetailed metadata for {sample_person}:")
        for key, value in sample_info["metadata"].items():
            print(f"  {key}: {value}")


def demo_edge_cases():
    """Demonstrate handling of edge cases."""

    print("\n" + "=" * 60)
    print("DEMO 4: Edge Cases")
    print("=" * 60)

    # Test case 1: Single person
    print("\n--- Single Person ---")
    single_person_data = [
        {"person_id": "person_001", "bbox": [100, 100, 80, 160]},
        {"person_id": "person_001", "bbox": [105, 105, 75, 155]},
    ]

    results = run_size_based_analysis(single_person_data)
    for person_id, info in results.items():
        print(
            f"{person_id}: {info['label']} (normalized_height={info['metadata']['normalized_height']:.2f})"
        )

    # Test case 2: All same height
    print("\n--- All Same Height ---")
    same_height_data = [
        {"person_id": "person_001", "bbox": [100, 100, 80, 120]},
        {"person_id": "person_002", "bbox": [200, 100, 75, 120]},
        {"person_id": "person_003", "bbox": [300, 100, 85, 120]},
    ]

    results = run_size_based_analysis(same_height_data)
    for person_id, info in results.items():
        print(
            f"{person_id}: {info['label']} (normalized_height={info['metadata']['normalized_height']:.2f})"
        )

    # Test case 3: Missing/invalid bounding boxes
    print("\n--- Invalid Bounding Boxes ---")
    invalid_bbox_data = [
        {"person_id": "person_001", "bbox": [100, 100, 80, 160]},  # Valid
        {"person_id": "person_001", "bbox": []},  # Invalid
        {"person_id": "person_002", "bbox": [200, 100]},  # Invalid
        {"person_id": "person_003"},  # Missing bbox
    ]

    results = run_size_based_analysis(invalid_bbox_data)
    print(f"Successfully analyzed {len(results)} persons despite invalid data")
    for person_id, info in results.items():
        print(f"{person_id}: {info['label']}")


def main():
    """Run all demo scenarios."""

    print("Size-Based Person Analysis - Demo Script")
    print("This demonstrates the first version of automated person analysis")
    print("using simple size-based inference to classify adults vs children.")

    # Run built-in demo first
    print("\n" + "=" * 60)
    print("BUILT-IN DEMO: Basic Example")
    print("=" * 60)
    demo_size_based_analysis()

    # Run custom demos
    demo_basic_analysis()
    demo_custom_threshold()
    demo_analyzer_class()
    demo_edge_cases()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print(
        "1. Size-based inference classifies persons as adult/child based on relative heights"
    )
    print("2. Height threshold (default 0.4) determines the classification boundary")
    print("3. All heights are normalized relative to the tallest person in the video")
    print(
        "4. The analyzer handles edge cases gracefully (single person, invalid data, etc.)"
    )
    print("5. Results include confidence scores and detailed reasoning metadata")

    print("\nNext steps:")
    print("- Integrate with person tracking pipeline for real video analysis")
    print("- Combine with position-based and temporal consistency filtering")
    print("- Test on real parent-child interaction videos")


if __name__ == "__main__":
    main()
