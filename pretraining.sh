#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH --job-name=ddp_single_node
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=1                      # One task (master launcher)
#SBATCH --cpus-per-task=4              # 4 CPU threads for the task
#SBATCH --gres=gpu:4                   # Request 4 GPUs
#SBATCH --mem=40GB                     # Total memory
#SBATCH -t 24:00:00                    # 24-hour wall time

# Load your environment
module load miniforge/24.3.0-0
source activate torch  # Replace with your conda env

echo "======== Starting Single-Node Multi-GPU Training ========"
torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
  -m src.pretraining.pretraining_main "$@"
