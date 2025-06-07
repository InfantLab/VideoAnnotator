#!/usr/bin/env python3
"""
Process video files to extract facial data for the BabyJokes project.
This script provides a command-line interface for the facial data extraction functionality.
"""

import os
import sys
import argparse
import pandas as pd
import time
import logging
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.processors.face_processor import process_video_faces, get_facial_stats
from src.processors.video_processor import get_video_metadata
from src.utils.io_utils import getProcessedVideos, saveProcessedVideos, get_stem_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"face_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

def process_videos(videos, output_dir, options):
    """
    Process multiple videos to extract facial data.
    
    Args:
        videos (list): List of video file paths
        output_dir (str): Directory to save results
        options (dict): Dictionary of processing options
    """
    # Get processed videos dataframe
    processedvideos = getProcessedVideos(output_dir)
    
    for video_path in videos:
        basename = os.path.basename(video_path)
        stemname = get_stem_name(video_path)
        logging.info(f"Processing video: {basename}")
        
        # Check if video is already in processedvideos
        row_exists = basename in processedvideos["VideoID"].values
        
        if row_exists:
            row = processedvideos.loc[processedvideos["VideoID"] == basename].iloc[0].to_dict()
        else:
            # Get video metadata
            metadata = get_video_metadata(video_path)
            if metadata is None:
                logging.error(f"Error opening video: {basename}")
                continue
            
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
            saveProcessedVideos(processedvideos, output_dir)
            
            # Get index of the new row
            idx = processedvideos.loc[processedvideos["VideoID"] == basename].index[0]
        
        # Check if we should process this video
        if options["force"] or pd.isna(processedvideos.loc[processedvideos["VideoID"] == basename, "Faces.file"].values[0]):
            try:
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
                    skip_frames=options["skip_frames"],
                    backend=options["backend"],
                    force_process=options["force"]
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
                
                logging.info(f"Successfully processed facial data for {basename}")
                
                # Generate and print statistics if requested
                if options["stats"] and faces_results.get("faces"):
                    faces_df = pd.read_csv(faces_results["faces"])
                    stats = get_facial_stats(faces_df)
                    logging.info(f"Facial statistics for {basename}:")
                    for key, value in stats.items():
                        logging.info(f"  {key}: {value}")
                
            except Exception as e:
                logging.exception(f"Error processing facial data for {basename}: {str(e)}")
        else:
            logging.info(f"Facial data already processed for {basename}")

def main():
    parser = argparse.ArgumentParser(description="Process video files to extract facial data")
    
    # Input/output options
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory of videos or path to a single video")
    parser.add_argument("--output", "-o", type=str, default="./data", help="Output directory for processed data")
    
    # Processing options
    parser.add_argument("--backend", "-b", type=str, default="ssd", 
                        choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet'],
                        help="DeepFace detector backend")
    parser.add_argument("--skip-frames", "-s", type=int, default=0, help="Number of frames to skip between processing")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing of existing files")
    parser.add_argument("--stats", action="store_true", help="Generate and display facial statistics")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    options = {
        "backend": args.backend,
        "skip_frames": args.skip_frames,
        "force": args.force,
        "stats": args.stats
    }
    
    # Get list of videos to process
    videos = []
    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                videos.append(os.path.join(args.input, file))
        logging.info(f"Found {len(videos)} videos to process in directory: {args.input}")
    else:
        if os.path.exists(args.input) and args.input.lower().endswith((".mp4", ".avi", ".mov")):
            videos.append(args.input)
            logging.info(f"Processing single video: {args.input}")
        else:
            logging.error(f"Invalid input: {args.input}")
            return 1
    
    if not videos:
        logging.error("No video files found to process")
        return 1
    
    # Process videos
    process_videos(videos, args.output, options)
    
    logging.info("Facial data processing complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
