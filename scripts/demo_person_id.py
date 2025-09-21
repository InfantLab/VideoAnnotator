#!/usr/bin/env python3
"""
PersonID Implementation Demo

This script demonstrates the PersonID tracking and labeling system with realistic examples.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))

from person_identity import PersonIdentityManager, normalize_person_label, get_available_labels
from automatic_labeling import AutomaticPersonLabeler, infer_person_labels_from_tracks


def create_realistic_video_data():
    """Create realistic video tracking data for demonstration."""
    
    # Simulate a parent-child interaction video with realistic bounding boxes
    # Parent: larger person, sometimes in background
    # Child: smaller person, often in center of frame
    
    annotations = []
    annotation_id = 1
    
    # Generate frames over 30 seconds (5 fps = 150 frames)
    for frame_num in range(1, 151):
        timestamp = frame_num / 5.0  # 5 fps
        
        # Parent (track_id=1): Larger bounding box, varies position
        if frame_num % 3 == 0:  # Present in ~2/3 of frames
            parent_x = 50 + (frame_num % 100)  # Moves around
            parent_y = 30 + (frame_num % 50)
            parent_bbox = [parent_x, parent_y, 90, 180]  # Large person
            
            annotations.append({
                "id": annotation_id,
                "image_id": f"demo_video_frame_{frame_num:03d}",
                "category_id": 1,
                "bbox": parent_bbox,
                "area": parent_bbox[2] * parent_bbox[3],
                "iscrowd": 0,
                "score": 0.85 + (frame_num % 10) * 0.01,
                "track_id": 1,
                "timestamp": timestamp,
                "frame_number": frame_num
            })
            annotation_id += 1
        
        # Child (track_id=2): Smaller bounding box, often centered
        if frame_num % 2 == 0:  # Present in ~1/2 of frames
            # Child tends to be more centered
            child_x = 200 + (frame_num % 40) - 20  # More centered movement
            child_y = 120 + (frame_num % 30) - 15
            child_bbox = [child_x, child_y, 45, 90]  # Small person
            
            annotations.append({
                "id": annotation_id,
                "image_id": f"demo_video_frame_{frame_num:03d}",
                "category_id": 1,
                "bbox": child_bbox,
                "area": child_bbox[2] * child_bbox[3],
                "iscrowd": 0,
                "score": 0.80 + (frame_num % 8) * 0.01,
                "track_id": 2,
                "timestamp": timestamp,
                "frame_number": frame_num
            })
            annotation_id += 1
            
        # Occasionally, add a third person (track_id=3) - another adult
        if frame_num > 100 and frame_num % 10 == 0:
            other_x = 400 + (frame_num % 60)
            other_y = 40 + (frame_num % 40)
            other_bbox = [other_x, other_y, 85, 175]  # Another large person
            
            annotations.append({
                "id": annotation_id,
                "image_id": f"demo_video_frame_{frame_num:03d}",
                "category_id": 1,
                "bbox": other_bbox,
                "area": other_bbox[2] * other_bbox[3],
                "iscrowd": 0,
                "score": 0.78,
                "track_id": 3,
                "timestamp": timestamp,
                "frame_number": frame_num
            })
            annotation_id += 1
    
    return annotations


def demo_person_identity_tracking():
    """Demonstrate person identity tracking and automatic labeling."""
    
    print("PersonID Implementation Demo")
    print("=" * 60)
    print("Simulating a parent-child interaction video...\n")
    
    # Create realistic video data
    annotations = create_realistic_video_data()
    print(f"Generated {len(annotations)} person detections across 150 frames")
    
    # Initialize PersonIdentityManager
    video_id = "demo_video"
    manager = PersonIdentityManager(video_id, "semantic")
    print(f"Initialized PersonIdentityManager for '{video_id}'")
    
    # Add person_id to all annotations (simulate pipeline processing)
    for annotation in annotations:
        track_id = annotation.get('track_id')
        if track_id is not None:
            person_id = manager.register_track(track_id)
            annotation['person_id'] = person_id
    
    print(f"Registered {len(manager.get_all_person_ids())} unique persons")
    
    # Analyze the tracking data
    print("\n" + "=" * 60)
    print("TRACKING ANALYSIS")
    print("=" * 60)
    
    person_stats = {}
    for annotation in annotations:
        person_id = annotation.get('person_id')
        if person_id:
            if person_id not in person_stats:
                person_stats[person_id] = {
                    'detections': 0,
                    'avg_height': 0,
                    'heights': [],
                    'center_positions': []
                }
            
            stats = person_stats[person_id]
            stats['detections'] += 1
            
            bbox = annotation.get('bbox', [])
            if len(bbox) >= 4:
                height = bbox[3]
                stats['heights'].append(height)
                
                # Calculate center position (normalized to 0-1)
                center_x = (bbox[0] + bbox[2]/2) / 640  # Assume 640px width
                center_y = (bbox[1] + bbox[3]/2) / 480  # Assume 480px height
                stats['center_positions'].append((center_x, center_y))
    
    # Calculate averages
    for person_id, stats in person_stats.items():
        if stats['heights']:
            stats['avg_height'] = sum(stats['heights']) / len(stats['heights'])
        if stats['center_positions']:
            avg_center_x = sum(pos[0] for pos in stats['center_positions']) / len(stats['center_positions'])
            avg_center_y = sum(pos[1] for pos in stats['center_positions']) / len(stats['center_positions'])
            stats['avg_center'] = (avg_center_x, avg_center_y)
            stats['center_bias'] = 1.0 - ((avg_center_x - 0.5)**2 + (avg_center_y - 0.5)**2)**0.5
    
    # Display analysis
    for person_id, stats in person_stats.items():
        print(f"\n{person_id}:")
        print(f"  Detections: {stats['detections']}")
        print(f"  Average height: {stats['avg_height']:.1f}px")
        if 'avg_center' in stats:
            print(f"  Average position: ({stats['avg_center'][0]:.2f}, {stats['avg_center'][1]:.2f})")
            print(f"  Center bias: {stats['center_bias']:.3f}")
    
    # Apply automatic labeling
    print("\n" + "=" * 60)
    print("AUTOMATIC LABELING")
    print("=" * 60)
    
    # Configure automatic labeling for demo
    auto_config = {
        "size_based": {
            "enabled": True,
            "height_threshold": 120,  # Absolute height threshold for demo
            "confidence": 0.8,
            "adult_label": "parent",
            "child_label": "infant"
        },
        "position_based": {
            "enabled": True,
            "center_bias_threshold": 0.6,
            "confidence": 0.7,
            "primary_label": "infant",
            "secondary_label": "parent"
        },
        "temporal_consistency": {
            "enabled": True,
            "min_detections": 20,
            "consistency_threshold": 0.7
        }
    }
    
    labeler = AutomaticPersonLabeler(auto_config)
    automatic_labels = labeler.label_persons_automatic(annotations, annotations)
    
    print("Automatic labeling results:")
    for person_id, label_info in automatic_labels.items():
        print(f"  {person_id}: '{label_info['label']}' "
              f"(confidence={label_info['confidence']:.2f}, method={label_info['method']})")
        
        # Apply labels to identity manager
        manager.set_person_label(
            person_id,
            label_info['label'],
            label_info['confidence'],
            label_info['method']
        )
    
    # Update annotations with labels
    labeled_count = 0
    for annotation in annotations:
        person_id = annotation.get('person_id')
        if person_id:
            label_info = manager.get_person_label(person_id)
            if label_info:
                annotation['person_label'] = label_info['label']
                annotation['label_confidence'] = label_info['confidence']
                annotation['labeling_method'] = label_info['method']
                labeled_count += 1
    
    print(f"\nLabeled {labeled_count} out of {len(annotations)} annotations")
    
    # Demonstrate manual labeling override
    print("\n" + "=" * 60)
    print("MANUAL LABELING OVERRIDE")
    print("=" * 60)
    
    # Let's say we want to manually label the third person as "teacher"
    all_person_ids = list(person_stats.keys())
    if len(all_person_ids) >= 3:
        third_person = all_person_ids[2]
        print(f"Manually labeling {third_person} as 'teacher'...")
        
        manager.set_person_label(third_person, "teacher", 1.0, "manual")
        
        # Update annotations
        for annotation in annotations:
            if annotation.get('person_id') == third_person:
                label_info = manager.get_person_label(third_person)
                annotation['person_label'] = label_info['label']
                annotation['label_confidence'] = label_info['confidence']
                annotation['labeling_method'] = label_info['method']
    
    # Show final labeling summary
    print("\n" + "=" * 60)
    print("FINAL LABELING SUMMARY")
    print("=" * 60)
    
    label_summary = {}
    for annotation in annotations:
        person_id = annotation.get('person_id')
        person_label = annotation.get('person_label', 'unlabeled')
        
        if person_label not in label_summary:
            label_summary[person_label] = {}
        if person_id not in label_summary[person_label]:
            label_summary[person_label][person_id] = 0
        label_summary[person_label][person_id] += 1
    
    for label, persons in label_summary.items():
        print(f"\n{label.upper()}:")
        for person_id, count in persons.items():
            confidence = "N/A"
            method = "N/A"
            label_info = manager.get_person_label(person_id)
            if label_info:
                confidence = f"{label_info['confidence']:.2f}"
                method = label_info['method']
            
            print(f"  {person_id}: {count} detections (confidence={confidence}, method={method})")
    
    # Save results for inspection
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save person tracks
    tracks_file = "demo_person_tracks.json"
    detection_summary = {
        "total_detections": len(annotations),
        "unique_persons": len(manager.get_all_person_ids()),
        "labeled_persons": len([p for p in manager.get_all_person_ids() 
                              if manager.get_person_label(p)])
    }
    manager.save_person_tracks(tracks_file, detection_summary)
    print(f"Saved person tracks: {tracks_file}")
    
    # Save sample annotations
    sample_annotations_file = "demo_annotations_sample.json"
    sample_data = {
        "video_info": {
            "video_id": video_id,
            "total_frames": 150,
            "fps": 5,
            "duration": 30.0
        },
        "sample_annotations": annotations[:10],  # First 10 annotations
        "summary": detection_summary
    }
    
    with open(sample_annotations_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Saved sample annotations: {sample_annotations_file}")
    
    print(f"\n🎉 Demo complete! PersonID system successfully:")
    print(f"   ✓ Tracked {len(manager.get_all_person_ids())} unique persons")
    print(f"   ✓ Generated {len(annotations)} person detections")
    print(f"   ✓ Applied automatic labeling")
    print(f"   ✓ Supported manual label overrides")
    print(f"   ✓ Maintained consistent IDs across {len(set(a['frame_number'] for a in annotations))} frames")


if __name__ == "__main__":
    try:
        demo_person_identity_tracking()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
