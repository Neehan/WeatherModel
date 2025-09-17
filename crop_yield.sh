#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=crop_yield_training
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=1                      # One task (master launcher)
#SBATCH --cpus-per-task=4              # 4 CPU threads for the task
#SBATCH --gres=gpu:1                   # Request 4 GPUs
#SBATCH --mem=40GB                     # Total memory
#SBATCH -t 24:00:00                    # 24-hour wall time
# Load your environment
module load miniforge/24.3.0-0
export TRANSFORMERS_NO_TORCHVISION=1

echo "======== Starting Crop Yield Training ========"
python -m src.crop_yield.yield_main "$@"
