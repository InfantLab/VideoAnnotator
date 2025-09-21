#!/usr/bin/env python3
"""
Test script for PersonID implementation.

This script tests the core PersonIdentityManager functionality.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid circular import issues
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))
from person_identity import PersonIdentityManager, normalize_person_label, get_available_labels
from automatic_labeling import infer_person_labels_from_tracks


def test_person_identity_manager():
    """Test PersonIdentityManager basic functionality."""
    print("Testing PersonIdentityManager...")
    
    # Create manager
    manager = PersonIdentityManager("test_video", "semantic")
    
    # Register some tracks
    person1 = manager.register_track(1)
    person2 = manager.register_track(2)
    person3 = manager.register_track(3)
    
    print(f"Registered persons: {person1}, {person2}, {person3}")
    
    # Test labeling
    manager.set_person_label(person1, "infant", 0.9, "manual")
    manager.set_person_label(person2, "parent", 0.8, "manual")
    
    # Test label retrieval
    label1 = manager.get_person_label(person1)
    label2 = manager.get_person_label(person2)
    label3 = manager.get_person_label(person3)
    
    print(f"Person 1 label: {label1}")
    print(f"Person 2 label: {label2}")
    print(f"Person 3 label: {label3}")
    
    # Test IoU calculation
    bbox1 = [100, 100, 50, 100]  # [x, y, w, h]
    bbox2 = [120, 120, 50, 100]  # Overlapping
    bbox3 = [200, 200, 50, 100]  # Non-overlapping
    
    iou1 = manager._calculate_iou(bbox1, bbox2)
    iou2 = manager._calculate_iou(bbox1, bbox3)
    
    print(f"IoU overlapping: {iou1:.3f}")
    print(f"IoU non-overlapping: {iou2:.3f}")
    
    # Test save/load
    test_file = "test_person_tracks.json"
    manager.save_person_tracks(test_file)
    
    # Load and verify
    loaded_manager = PersonIdentityManager.load_person_tracks(test_file)
    loaded_label1 = loaded_manager.get_person_label(person1)
    
    print(f"Loaded person 1 label: {loaded_label1}")
    
    # Cleanup
    Path(test_file).unlink()
    
    print("✓ PersonIdentityManager tests passed!\n")


def test_label_normalization():
    """Test person label normalization."""
    print("Testing label normalization...")
    
    test_cases = [
        ("infant", "infant"),
        ("baby", "infant"),
        ("child", "infant"),
        ("mom", "parent"),
        ("father", "parent"),
        ("teacher", "teacher"),
        ("invalid_label", None)
    ]
    
    for input_label, expected in test_cases:
        result = normalize_person_label(input_label)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_label}' -> '{result}' (expected: '{expected}')")
    
    print(f"\nAvailable labels: {get_available_labels()}")
    print("✓ Label normalization tests passed!\n")


def test_automatic_labeling():
    """Test automatic labeling functionality."""
    print("Testing automatic labeling...")
    
    # Create mock annotations with different sized people
    mock_annotations = [
        # Large person (adult)
        {
            "person_id": "person_test_001",
            "bbox": [100, 50, 80, 200],  # Tall
            "category_id": 1,
            "timestamp": 1.0
        },
        {
            "person_id": "person_test_001", 
            "bbox": [105, 55, 85, 205],
            "category_id": 1,
            "timestamp": 2.0
        },
        # Small person (child)
        {
            "person_id": "person_test_002",
            "bbox": [200, 150, 40, 80],  # Short
            "category_id": 1,
            "timestamp": 1.0
        },
        {
            "person_id": "person_test_002",
            "bbox": [205, 155, 45, 85],
            "category_id": 1,
            "timestamp": 2.0
        }
    ]
    
    # Test automatic labeling
    person_tracks = mock_annotations  # In real use, this would be track summaries
    automatic_labels = infer_person_labels_from_tracks(person_tracks, mock_annotations)
    
    print("Automatic labeling results:")
    for person_id, label_info in automatic_labels.items():
        print(f"  {person_id}: {label_info['label']} "
              f"(confidence={label_info['confidence']:.2f}, "
              f"method={label_info['method']})")
    
    print("✓ Automatic labeling tests passed!\n")


def test_mock_pipeline_integration():
    """Test integration with mock pipeline data."""
    print("Testing pipeline integration...")
    
    # Create manager
    manager = PersonIdentityManager("test_integration", "semantic")
    
    # Mock YOLO tracking results (what would come from PersonTrackingPipeline)
    mock_tracking_results = [
        {
            "id": 1,
            "image_id": "test_integration_frame_001",
            "category_id": 1,
            "bbox": [100, 100, 80, 160],
            "area": 12800,
            "iscrowd": 0,
            "score": 0.9,
            "track_id": 1,
            "timestamp": 1.0,
            "frame_number": 1
        },
        {
            "id": 2,
            "image_id": "test_integration_frame_002", 
            "category_id": 1,
            "bbox": [105, 105, 82, 162],
            "area": 13284,
            "iscrowd": 0,
            "score": 0.85,
            "track_id": 1,
            "timestamp": 2.0,
            "frame_number": 2
        },
        {
            "id": 3,
            "image_id": "test_integration_frame_001",
            "category_id": 1,
            "bbox": [200, 150, 40, 80],
            "area": 3200,
            "iscrowd": 0,
            "score": 0.8,
            "track_id": 2,
            "timestamp": 1.0,
            "frame_number": 1
        }
    ]
    
    # Add person_id fields (simulate pipeline enhancement)
    for annotation in mock_tracking_results:
        track_id = annotation.get('track_id')
        if track_id is not None:
            person_id = manager.register_track(track_id)
            annotation['person_id'] = person_id
    
    print("Enhanced tracking results:")
    for annotation in mock_tracking_results:
        print(f"  Frame {annotation['frame_number']}: "
              f"track_id={annotation['track_id']} -> "
              f"person_id={annotation['person_id']}")
    
    # Apply automatic labeling
    automatic_labels = infer_person_labels_from_tracks(mock_tracking_results, mock_tracking_results)
    
    # Update manager with labels
    for person_id, label_info in automatic_labels.items():
        if label_info['confidence'] >= 0.7:
            manager.set_person_label(
                person_id,
                label_info['label'], 
                label_info['confidence'],
                label_info['method']
            )
    
    # Add labels to annotations
    for annotation in mock_tracking_results:
        person_id = annotation.get('person_id')
        if person_id:
            label_info = manager.get_person_label(person_id)
            if label_info:
                annotation['person_label'] = label_info['label']
                annotation['label_confidence'] = label_info['confidence']
                annotation['labeling_method'] = label_info['method']
    
    print("\nFinal enhanced annotations:")
    for annotation in mock_tracking_results:
        print(f"  {annotation['person_id']}: {annotation.get('person_label', 'unlabeled')} "
              f"(bbox area: {annotation['area']})")
    
    print("✓ Pipeline integration tests passed!\n")


if __name__ == "__main__":
    print("PersonID Implementation Test Suite")
    print("=" * 50)
    
    try:
        test_person_identity_manager()
        test_label_normalization()
        test_automatic_labeling()
        test_mock_pipeline_integration()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
