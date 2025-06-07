#!/usr/bin/env python3
import os
import argparse
import subprocess

def create_slurm_script(videos_in, data_out):
    """Create a SLURM script for video understanding job submission to the HPC."""
    script = f"""#!/bin/bash
#SBATCH --job-name=video_understanding
#SBATCH --output=video_understanding_%j.out
#SBATCH --error=video_understanding_%j.err
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
from src.main import understand_videos

understand_videos(
    videos_in='{videos_in}',
    data_out='{data_out}'
)
"
"""
    script_path = os.path.join(os.getcwd(), 'scripts', 'video_understanding_job.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description='Submit video understanding job to HPC')
    parser.add_argument('--videos_in', required=True, help='Directory containing input videos')
    parser.add_argument('--data_out', required=True, help='Directory to save processed data')
    
    args = parser.parse_args()
    
    # Create SLURM script
    script_path = create_slurm_script(args.videos_in, args.data_out)
    
    # Submit the job
    subprocess.run(['sbatch', script_path])
    print(f"Job submitted with script: {script_path}")

if __name__ == "__main__":
    main()
