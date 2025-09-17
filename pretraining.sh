#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=pretrain
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=1                      # One task (master launcher)
#SBATCH --cpus-per-task=4              # 4 CPU threads for the task
#SBATCH --gres=gpu:h100:4                   # Request 4 GPUs
#SBATCH --mem=40GB                     # Total memory
#SBATCH -t 24:00:00                    # 24-hour wall time

# Load your environment
module load miniforge/24.3.0-0

# no torch vision dependencies
export TRANSFORMERS_NO_TORCHVISION=1

echo "======== Starting Single-Node Multi-GPU Training ========"

# Extract model names from arguments
models=()
other_args=()
collecting_models=true

# Parse arguments: first args are model names, then -- args are parameters
for arg in "$@"; do
    if [[ $arg == --* ]]; then
        collecting_models=false
        other_args+=("$arg")
    elif $collecting_models; then
        models+=("$arg")
    else
        other_args+=("$arg")
    fi
done

# If no models specified, default to weatherformer
if [ ${#models[@]} -eq 0 ]; then
    models=("weatherformer")
fi

echo "Training models: ${models[*]}"
echo "Other arguments: ${other_args[*]}"

# Train each model
for model in "${models[@]}"; do
    echo "======== Training model: $model ========"
    torchrun \
      --nnodes=1 \
      --nproc-per-node=4 \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=localhost:29500 \
      -m src.pretraining.pretraining_main --model "$model" "${other_args[@]}"
    
    echo "======== Completed training for model: $model ========"
done

echo "======== All models training completed ========"