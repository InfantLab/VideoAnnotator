#!/usr/bin/env python3
"""Test face detection with different parameters."""

import cv2
import numpy as np
from pathlib import Path

def test_opencv_face_detection(video_path: str, min_face_sizes=[10, 20, 30, 50]):
    """Test OpenCV face detection with different minimum face sizes."""
    print(f"Testing face detection on: {video_path}")
    
    # Load cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Test detection on a few frames
    test_frames = [0, total_frames//4, total_frames//2, 3*total_frames//4]
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
            
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print(f"\nFrame {frame_num} ({width}x{height}):")
        
        for min_size in min_face_sizes:
            # Try with more aggressive parameters
            for scale_factor, min_neighbors in [(1.05, 3), (1.1, 5), (1.3, 8)]:
                face_rects = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_size, min_size),
                )
                if len(face_rects) > 0:
                    print(f"  Min size {min_size}px, scale {scale_factor}, neighbors {min_neighbors}: {len(face_rects)} faces detected")
                    for i, (x, y, w, h) in enumerate(face_rects):
                        print(f"    Face {i}: ({x},{y}) {w}x{h} pixels")
                    break
            else:
                print(f"  Min size {min_size}px: 0 faces detected with all parameter combinations")
    
    cap.release()

if __name__ == "__main__":
    video_path = "demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4"
    test_opencv_face_detection(video_path)
