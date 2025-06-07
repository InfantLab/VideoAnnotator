#!/usr/bin/env python3
"""
Batch process video files for the BabyJokes project.
This script is designed to run on HPC systems and can process multiple videos in parallel.
"""

import os
import sys
import argparse
import pandas as pd
import time
from pathlib import Path
import logging
import concurrent.futures
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.processors.video_processor import videotokeypoints, get_video_metadata
from src.processors.keypoint_processor import process_keypoints_for_modeling
from src.processors.face_processor import process_video_faces
from src.processors.audio_processor import process_audio
from src.processors.object_processor import process_video_objects
from src.processors.video_understanding import extract_video_understanding
from src.utils.io_utils import getProcessedVideos, saveProcessedVideos, get_stem_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"batch_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

def process_single_video(video_path, output_dir, options):
    """
    Process a single video with all enabled features.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save results
        options (dict): Dictionary of processing options
    
    Returns:
        dict: Dictionary of processing results
    """
    try:
        basename = os.path.basename(video_path)
        stemname = get_stem_name(video_path)
        logging.info(f"Processing video: {basename}")
        
        results = {}
        
        # Get or initialize the tracking DataFrame
        processedvideos = getProcessedVideos(output_dir)
        row_exists = basename in processedvideos["VideoID"].values
        
        if row_exists:
            row = processedvideos.loc[processedvideos["VideoID"] == basename].iloc[0].to_dict()
        else:
            # Get video metadata
            metadata = get_video_metadata(video_path)
            if metadata is None:
                logging.error(f"Error opening video: {basename}")
                return None
            
            # Create new row
            row = {
                "VideoID": basename,
                "Width": metadata.get("Width", 0),
                "Height": metadata.get("Height", 0),
                "FPS": metadata.get("FPS", 0),
                "Duration": metadata.get("Duration", 0),
                "TotalFrames": metadata.get("TotalFrames", 0),
            }
            
            # Add row to processedvideos
            processedvideos = pd.concat([processedvideos, pd.DataFrame([row])], ignore_index=True)
        
        # Process keypoints if enabled
        if options.get("keypoints", False):
            if options.get("force", False) or pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Keypoints.file"].values[0]):
                logging.info(f"Extracting keypoints from {basename}")
                
                # Extract keypoints
                model_path = options.get("pose_model", "yolov8n-pose.pt")
                from ultralytics import YOLO
                model = YOLO(model_path)
                
                keypointsdf = videotokeypoints(model, video_path, track=True)
                
                # Save keypoints
                keypointspath = os.path.join(output_dir, f"{stemname}.csv")
                keypointsdf.to_csv(keypointspath, index=False)
                
                # Update row
                idx = processedvideos.loc[processedvideos["VideoID"] == basename].index[0]
                processedvideos.at[idx, "Keypoints.file"] = keypointspath
                processedvideos.at[idx, "Keypoints.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                # Process normalized keypoints
                normedkeypointsdf = process_keypoints_for_modeling(keypointsdf, row["Height"], row["Width"])
                normedkeypointspath = os.path.join(output_dir, f"{stemname}_normed.csv")
                normedkeypointsdf.to_csv(normedkeypointspath, index=False)
                processedvideos.at[idx, "Keypoints.normed"] = normedkeypointspath
                
                # Save progress
                saveProcessedVideos(processedvideos, output_dir)
                
                results["keypoints"] = {
                    "raw": keypointspath,
                    "normed": normedkeypointspath
                }
            else:
                logging.info(f"Keypoints already extracted for {basename}")
        
        # Process audio if enabled
        if options.get("audio", False):
            if options.get("force", False) or pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Audio.file"].values[0]):
                logging.info(f"Processing audio from {basename}")
                
                # Process audio
                audio_results = process_audio(
                    video_path, 
                    output_dir, 
                    enable_whisper=options.get("whisper", True),
                    enable_diarization=options.get("diarization", True),
                    enable_f0=options.get("f0", True),
                    enable_laughter=options.get("laughter", True),
                    force_process=options.get("force", False)
                )
                
                # Update row
                idx = processedvideos.loc[processedvideos["VideoID"] == basename].index[0]
                if audio_results.get("audio"):
                    processedvideos.at[idx, "Audio.file"] = audio_results["audio"]
                    processedvideos.at[idx, "Audio.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                if audio_results.get("transcript"):
                    processedvideos.at[idx, "Speech.file"] = audio_results["transcript"]
                    processedvideos.at[idx, "Speech.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                if audio_results.get("diarization"):
                    processedvideos.at[idx, "Diary.file"] = audio_results["diarization"]
                    processedvideos.at[idx, "Diary.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                if audio_results.get("f0"):
                    processedvideos.at[idx, "F0.file"] = audio_results["f0"]
                    processedvideos.at[idx, "F0.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                if audio_results.get("laughter"):
                    processedvideos.at[idx, "Laughter.file"] = audio_results["laughter"]
                    processedvideos.at[idx, "Laughter.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                # Save progress
                saveProcessedVideos(processedvideos, output_dir)
                
                results["audio"] = audio_results
            else:
                logging.info(f"Audio already processed for {basename}")
        
        # Process faces if enabled
        if options.get("faces", False):
            if options.get("force", False) or pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Faces.file"].values[0]):
                logging.info(f"Processing faces from {basename}")
                
                # Get poses dataframe if needed for matching
                poses_df = None
                if not pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Keypoints.normed"].values[0]):
                    poses_df = pd.read_csv(processedvideos.loc[processedvideos["VideoID"] == basename, "Keypoints.normed"].values[0])
                
                # Process faces
                faces_results = process_video_faces(
                    video_path, 
                    output_dir, 
                    video_metadata=row,
                    poses_df=poses_df,
                    skip_frames=options.get("skip_frames", 0),
                    backend=options.get("face_backend", "ssd"),
                    force_process=options.get("force", False)
                )
                
                # Update row
                idx = processedvideos.loc[processedvideos["VideoID"] == basename].index[0]
                if faces_results.get("faces"):
                    processedvideos.at[idx, "Faces.file"] = faces_results["faces"]
                    processedvideos.at[idx, "Faces.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                if faces_results.get("normed"):
                    processedvideos.at[idx, "Faces.normed"] = faces_results["normed"]
                
                if faces_results.get("matched"):
                    processedvideos.at[idx, "Faces.matched"] = faces_results["matched"]
                
                # Save progress
                saveProcessedVideos(processedvideos, output_dir)
                
                results["faces"] = faces_results
            else:
                logging.info(f"Faces already processed for {basename}")
        
        # Process objects if enabled
        if options.get("objects", False):
            if options.get("force", False) or pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Objects.file"].values[0]):
                logging.info(f"Processing objects from {basename}")
                
                # Get poses dataframe if needed for matching
                poses_df = None
                if not pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Keypoints.normed"].values[0]):
                    poses_df = pd.read_csv(processedvideos.loc[processedvideos["VideoID"] == basename, "Keypoints.normed"].values[0])
                
                # Process objects
                objects_results = process_video_objects(
                    video_path, 
                    output_dir, 
                    video_metadata=row,
                    poses_df=poses_df,
                    confidence=options.get("object_confidence", 0.4),
                    model_path=options.get("object_model", "yolov8n.pt"),
                    force_process=options.get("force", False)
                )
                
                # Update row
                idx = processedvideos.loc[processedvideos["VideoID"] == basename].index[0]
                if objects_results.get("objects"):
                    processedvideos.at[idx, "Objects.file"] = objects_results["objects"]
                    processedvideos.at[idx, "Objects.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                if objects_results.get("normed"):
                    processedvideos.at[idx, "Objects.normed"] = objects_results["normed"]
                
                if objects_results.get("matched"):
                    processedvideos.at[idx, "Objects.matched"] = objects_results["matched"]
                
                # Save progress
                saveProcessedVideos(processedvideos, output_dir)
                
                results["objects"] = objects_results
            else:
                logging.info(f"Objects already processed for {basename}")
        
        # Process video understanding if enabled
        if options.get("understanding", False):
            if options.get("force", False) or pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Understanding.file"].values[0]):
                logging.info(f"Extracting understanding from {basename}")
                
                # Extract video understanding
                understanding_file = extract_video_understanding(video_path, output_dir)
                
                # Update row
                idx = processedvideos.loc[processedvideos["VideoID"] == basename].index[0]
                processedvideos.at[idx, "Understanding.file"] = understanding_file
                processedvideos.at[idx, "Understanding.when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                
                # Save progress
                saveProcessedVideos(processedvideos, output_dir)
                
                results["understanding"] = understanding_file
            else:
                logging.info(f"Understanding already extracted for {basename}")
        
        logging.info(f"Completed processing video: {basename}")
        return results
    
    except Exception as e:
        logging.exception(f"Error processing video {video_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Batch process video files for the BabyJokes project")
    
    # Input/output options
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory of videos or path to a single video")
    parser.add_argument("--output", "-o", type=str, default="data", help="Output directory for processed data")
    parser.add_argument("--metadata", "-m", type=str, help="Path to metadata Excel file")
    
    # Processing options
    parser.add_argument("--keypoints", action="store_true", help="Extract pose keypoints")
    parser.add_argument("--audio", action="store_true", help="Process audio")
    parser.add_argument("--faces", action="store_true", help="Extract facial data")
    parser.add_argument("--objects", action="store_true", help="Extract object data")
    parser.add_argument("--understanding", action="store_true", help="Extract video understanding")
    parser.add_argument("--all", action="store_true", help="Enable all processing options")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing of existing files")
    
    # Configuration options
    parser.add_argument("--pose-model", type=str, default="yolov8n-pose.pt", help="YOLOv8 pose model path")
    parser.add_argument("--object-model", type=str, default="yolov8n.pt", help="YOLOv8 object model path")
    parser.add_argument("--face-backend", type=str, default="ssd", help="DeepFace detector backend")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip frames for face detection")
    parser.add_argument("--object-confidence", type=float, default=0.4, help="Confidence threshold for object detection")
    
    # Audio options
    parser.add_argument("--no-whisper", action="store_true", help="Disable Whisper transcription")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--no-f0", action="store_true", help="Disable F0 extraction")
    parser.add_argument("--no-laughter", action="store_true", help="Disable laughter detection")
    
    # Parallel processing
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Prepare processing options
    options = {
        "keypoints": args.keypoints or args.all,
        "audio": args.audio or args.all,
        "faces": args.faces or args.all,
        "objects": args.objects or args.all,
        "understanding": args.understanding or args.all,
        "force": args.force,
        "pose_model": args.pose_model,
        "object_model": args.object_model,
        "face_backend": args.face_backend,
        "skip_frames": args.skip_frames,
        "object_confidence": args.object_confidence,
        "whisper": not args.no_whisper,
        "diarization": not args.no_diarization,
        "f0": not args.no_f0,
        "laughter": not args.no_laughter
    }
    
    # If no specific options are enabled, use keypoints as default
    if not any([args.keypoints, args.audio, args.faces, args.objects, args.understanding, args.all]):
        options["keypoints"] = True
        logging.info("No processing options specified, defaulting to keypoints extraction")
    
    # Get list of videos to process
    videos = []
    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                videos.append(os.path.join(args.input, file))
    else:
        if os.path.exists(args.input) and args.input.lower().endswith((".mp4", ".avi", ".mov")):
            videos.append(args.input)
    
    if not videos:
        logging.error("No video files found to process")
        return 1
    
    logging.info(f"Found {len(videos)} videos to process")
    
    # Process videos
    if args.workers > 1:
        logging.info(f"Processing with {args.workers} parallel workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, video, args.output, options): video for video in videos}
            for future in concurrent.futures.as_completed(futures):
                video = futures[future]
                try:
                    result = future.result()
                    if result:
                        logging.info(f"Successfully processed {os.path.basename(video)}")
                    else:
                        logging.error(f"Failed to process {os.path.basename(video)}")
                except Exception as e:
                    logging.exception(f"Exception processing {os.path.basename(video)}: {str(e)}")
    else:
        logging.info("Processing videos sequentially")
        for video in videos:
            result = process_single_video(video, args.output, options)
            if result:
                logging.info(f"Successfully processed {os.path.basename(video)}")
            else:
                logging.error(f"Failed to process {os.path.basename(video)}")
    
    logging.info("Batch processing complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
