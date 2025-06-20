#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=crop_grid
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=4                      # Four tasks (one per GPU)
#SBATCH --cpus-per-task=4              # 4 CPU threads per task
#SBATCH --gres=gpu:h100:4              # Request 4 GPUs
#SBATCH --mem=64GB                     # Total memory
#SBATCH -t 24:00:00                    # 24-hour wall time

# Load your environment
module load miniforge/24.3.0-0

echo "======== Starting Parallel Grid Search on 4 GPUs ========"

# Create output and log directories
mkdir -p data/grid_search
mkdir -p log

# Function to run experiment and log output
run_experiment() {
    local gpu_id=$1
    local model=$2
    local pretrained_flag=$3
    local log_file="logs/gpu${gpu_id}.log"
    
    echo "$(date): Starting ${model} ${pretrained_flag} on GPU ${gpu_id}" | tee -a "$log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python src/crop_yield/grid_search.py \
        --model "$model" \
        $pretrained_flag \
        --output-dir data/grid_search \
        >> "$log_file" 2>&1
    
    echo "$(date): Completed ${model} ${pretrained_flag} on GPU ${gpu_id}" | tee -a "$log_file"
}

# Clear previous logs
rm -f logs/gpu*.log

echo "Starting all 4 experiments in parallel..."

# Run all experiments in parallel, each on its own GPU with separate logging
echo "GPU 0: weatherformersinusoid (no pretraining)"
run_experiment 0 "weatherformersinusoid" "" &
PID1=$!

echo "GPU 1: weatherformersinusoid (with pretraining)"
run_experiment 1 "weatherformersinusoid" "--load-pretrained" &
PID2=$!

echo "GPU 2: weatherformermixture (no pretraining)"
run_experiment 2 "weatherformermixture" "" &
PID3=$!

echo "GPU 3: weatherformermixture (with pretraining)"
run_experiment 3 "weatherformermixture" "--load-pretrained" &
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
echo "You can monitor progress in real-time with: tail -f logs/gpu*.log"
wait $PID1 $PID2 $PID3 $PID4

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo "======== All Grid Search Experiments Completed ========"
echo "Results saved in data/grid_search/ directory:"
echo "- grid_search_weatherformersinusoid_not_pretrained.tsv"
echo "- grid_search_weatherformersinusoid_pretrained.tsv"
echo "- grid_search_weatherformermixture_not_pretrained.tsv"
echo "- grid_search_weatherformermixture_pretrained.tsv"
echo ""
echo "Logs saved in logs/ directory:"
echo "- logs/gpu0.log (weatherformersinusoid, no pretraining)"
echo "- logs/gpu1.log (weatherformersinusoid, with pretraining)"
echo "- logs/gpu2.log (weatherformermixture, no pretraining)"
echo "- logs/gpu3.log (weatherformermixture, with pretraining)" 