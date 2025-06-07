#!/usr/bin/env python3
import os
import argparse
import subprocess

def create_slurm_script(videos_in, data_out, metadata_file, force_metadata=False, force_process=False):
    """Create a SLURM script for job submission to the HPC."""
    script = f"""#!/bin/bash
#SBATCH --job-name=extract_movement
#SBATCH --output=extract_movement_%j.out
#SBATCH --error=extract_movement_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda3
source activate babyjokes  # Replace with your conda environment name

python -c "
import sys
sys.path.append('{os.getcwd()}')
from src.main import process_all_videos

process_all_videos(
    videos_in='{videos_in}',
    data_out='{data_out}',
    metadata_file='{metadata_file}',
    force_metadata={str(force_metadata).lower()},
    force_process={str(force_process).lower()}
)
"
"""
    script_path = os.path.join(os.getcwd(), 'scripts', 'extract_movement_job.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description='Submit keypoint extraction job to HPC')
    parser.add_argument('--videos_in', required=True, help='Directory containing input videos')
    parser.add_argument('--data_out', required=True, help='Directory to save processed data')
    parser.add_argument('--metadata_file', default="_LookitLaughter.test.xlsx", help='Metadata Excel file')
    parser.add_argument('--force_metadata', action='store_true', help='Force metadata extraction')
    parser.add_argument('--force_process', action='store_true', help='Force video processing')
    
    args = parser.parse_args()
    
    # Create SLURM script
    script_path = create_slurm_script(
        args.videos_in, 
        args.data_out, 
        args.metadata_file,
        args.force_metadata,
        args.force_process
    )
    
    # Submit the job
    subprocess.run(['sbatch', script_path])
    print(f"Job submitted with script: {script_path}")

if __name__ == "__main__":
    main()
