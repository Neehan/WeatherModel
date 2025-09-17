#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=crop_grid
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=4                      # Four tasks (one per GPU)
#SBATCH --cpus-per-task=4              # 4 CPU threads per task
#SBATCH --gres=gpu:l40s:4              # Request 4 GPUs
#SBATCH --mem=64GB                     # Total memory
#SBATCH -t 24:00:00                    # 24-hour wall time

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage (single model): $0 <model> <crop_type> [additional_python_args...]"
    echo "Usage (two models): $0 <model1> <model2> <crop_type> [additional_python_args...]"
    echo "Example (single): $0 weatherformer corn"
    echo "Example (single): $0 weatherformer corn --country argentina"
    echo "Example (single): $0 weatherformer corn --country usa --batch-size 128 --init-lr 0.001"
    echo "Example (two models): $0 weatherformer weatherformersinusoid corn --country argentina"
    echo "Available models: weatherbert, weatherformer, weatherformersinusoid, weatherformermixture, weatherautoencodermixture, weatherautoencoder, weatherautoencodersinusoid, simmtm, cnnrnn, gnnrnn, linear"
    echo "Available countries: usa, argentina (default: usa)"
    exit 1
fi

# Determine if we have 2 or 3+ arguments
if [ $# -eq 2 ] || [[ $3 != "corn" && $3 != "soybean" && $3 != "wheat" && $3 != "sunflower" ]]; then
    # Single model mode: model, crop_type, [extra_args...]
    MODEL1=$1
    MODEL2=$1  # Same model for both
    CROP_TYPE=$2
    EXTRA_ARGS="${@:3}"
    SINGLE_MODEL_MODE=true
else
    # Two model mode: model1, model2, crop_type, [extra_args...]
    MODEL1=$1
    MODEL2=$2
    CROP_TYPE=$3
    EXTRA_ARGS="${@:4}"
    SINGLE_MODEL_MODE=false
fi

# Validate model names
valid_models=("weatherbert" "weatherformer" "weatherformersinusoid" "weatherformermixture" "weatherautoencodermixture" "weatherautoencoder" "weatherautoencodersinusoid" "simmtm" "cnnrnn" "gnnrnn" "linear")
if [[ ! " ${valid_models[@]} " =~ " ${MODEL1} " ]]; then
    echo "Error: Invalid model1 '${MODEL1}'. Valid options: ${valid_models[@]}"
    exit 1
fi
if [[ ! " ${valid_models[@]} " =~ " ${MODEL2} " ]]; then
    echo "Error: Invalid model2 '${MODEL2}'. Valid options: ${valid_models[@]}"
    exit 1
fi

# Load your environment
module load miniforge/24.3.0-0

echo "======== Starting Parallel Grid Search ========"
if $SINGLE_MODEL_MODE; then
    echo "Single model mode: ${MODEL1} (pretrained vs not pretrained)"
    echo "Using 2 GPUs"
else
    echo "Two model mode: ${MODEL1} vs ${MODEL2}"
    echo "Using 4 GPUs"
fi
echo "Crop type: ${CROP_TYPE}"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra arguments: ${EXTRA_ARGS}"
fi

# Create output and log directories
mkdir -p data/grid_search
mkdir -p log

# rm -rf data/grid_search/*
# rm -rf log/gpu*.log
rm -rf data/trained_models/crop_yield/*

# Function to run experiment and log output
run_experiment() {
    local gpu_id=$1
    local model=$2
    local pretrained_flag=$3
    local log_file="log/gpu${gpu_id}.log"
    
    echo "$(date): Starting ${model} ${pretrained_flag} on GPU ${gpu_id}" | tee -a "$log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python -m src.crop_yield.grid_search \
        --model "$model" \
        --crop-type "$CROP_TYPE" \
        $pretrained_flag \
        --output-dir data/grid_search \
        $EXTRA_ARGS \
        >> "$log_file" 2>&1
    
    echo "$(date): Completed ${model} ${pretrained_flag} on GPU ${gpu_id}" | tee -a "$log_file"
}

# Clear previous logs
rm -f log/gpu*.log

if $SINGLE_MODEL_MODE; then
    echo "Starting 2 experiments in parallel (single model mode)..."
    
    # Run 2 experiments in parallel
    echo "GPU 0: ${MODEL1} (no pretraining)"
    run_experiment 0 "$MODEL1" "" &
    PID1=$!
    
    echo "GPU 1: ${MODEL1} (with pretraining)"
    run_experiment 1 "$MODEL1" "--load-pretrained" &
    PID2=$!
    
    # Store PIDs for monitoring
    PIDS=($PID1 $PID2)
    echo "Started processes with PIDs: ${PIDS[@]}"
    
    # Monitor progress
    monitor_progress() {
        while true; do
            sleep 300  # Check every 5 minutes
            echo "$(date): Progress check..."
            for i in "${!PIDS[@]}"; do
                if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    echo "  GPU $i: Still running (PID ${PIDS[$i]})"
                else
                    echo "  GPU $i: Completed (PID ${PIDS[$i]})"
                fi
            done
            
            # Check if any processes are still running
            running=false
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    running=true
                    break
                fi
            done
            
            if ! $running; then
                break
            fi
        done
    }
    
    # Start monitoring in background
    monitor_progress &
    MONITOR_PID=$!
    
    # Wait for all background jobs to complete
    echo "All experiments started. Waiting for completion..."
    echo "You can monitor progress in real-time with: tail -f log/gpu*.log"
    wait $PID1 $PID2
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null
    
    echo "======== All Grid Search Experiments Completed ========"
    echo "Results saved in data/grid_search/ directory:"
    echo "- grid_search_${MODEL1}_not_pretrained.tsv"
    echo "- grid_search_${MODEL1}_pretrained.tsv"
    echo ""
    echo "Logs saved in log/ directory:"
    echo "- log/gpu0.log (${MODEL1}, no pretraining)"
    echo "- log/gpu1.log (${MODEL1}, with pretraining)"
    
else
    echo "Starting all 4 experiments in parallel (two model mode)..."
    
    # Run all experiments in parallel, each on its own GPU with separate logging
    echo "GPU 0: ${MODEL1} (no pretraining)"
    run_experiment 0 "$MODEL1" "" &
    PID1=$!
    
    echo "GPU 1: ${MODEL1} (with pretraining)"
    run_experiment 1 "$MODEL1" "--load-pretrained" &
    PID2=$!
    
    echo "GPU 2: ${MODEL2} (no pretraining)"
    run_experiment 2 "$MODEL2" "" &
    PID3=$!
    
    echo "GPU 3: ${MODEL2} (with pretraining)"
    run_experiment 3 "$MODEL2" "--load-pretrained" &
    PID4=$!
    
    # Store all PIDs for monitoring
    PIDS=($PID1 $PID2 $PID3 $PID4)
    echo "Started processes with PIDs: ${PIDS[@]}"
    
    # Monitor progress
    monitor_progress() {
        while true; do
            sleep 300  # Check every 5 minutes
            echo "$(date): Progress check..."
            for i in "${!PIDS[@]}"; do
                if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    echo "  GPU $i: Still running (PID ${PIDS[$i]})"
                else
                    echo "  GPU $i: Completed (PID ${PIDS[$i]})"
                fi
            done
            
            # Check if any processes are still running
            running=false
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    running=true
                    break
                fi
            done
            
            if ! $running; then
                break
            fi
        done
    }
    
    # Start monitoring in background
    monitor_progress &
    MONITOR_PID=$!
    
    # Wait for all background jobs to complete
    echo "All experiments started. Waiting for completion..."
    echo "You can monitor progress in real-time with: tail -f log/gpu*.log"
    wait $PID1 $PID2 $PID3 $PID4
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null
    
    echo "======== All Grid Search Experiments Completed ========"
    echo "Results saved in data/grid_search/ directory:"
    echo "- grid_search_${MODEL1}_not_pretrained.tsv"
    echo "- grid_search_${MODEL1}_pretrained.tsv"
    echo "- grid_search_${MODEL2}_not_pretrained.tsv"
    echo "- grid_search_${MODEL2}_pretrained.tsv"
    echo ""
    echo "Logs saved in log/ directory:"
    echo "- log/gpu0.log (${MODEL1}, no pretraining)"
    echo "- log/gpu1.log (${MODEL1}, with pretraining)"
    echo "- log/gpu2.log (${MODEL2}, no pretraining)"
    echo "- log/gpu3.log (${MODEL2}, with pretraining)"
fi 