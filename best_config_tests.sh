#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=best_config_tests
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=4                      # Four tasks (one per GPU)
#SBATCH --cpus-per-task=4              # 4 CPU threads per task
#SBATCH --gres=gpu:l40s:4              # Request 4 GPUs
#SBATCH --mem=64GB                     # Total memory
#SBATCH -t 24:00:00                    # 12-hour wall time

# Initialize variables
MODEL=""
CROPS=()
COUNTRY="usa"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --crop)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                CROPS+=("$1")
                shift
            done
            ;;
        --country)
            COUNTRY="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ] || [ ${#CROPS[@]} -eq 0 ]; then
    echo "Usage: $0 --model <model> --crop <crop1> [crop2] [--country <country>] [additional_python_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 --model weatherformer --crop corn"
    echo "  $0 --model weatherformer --crop corn soybean"
    echo "  $0 --model weatherformer --crop corn --country usa"
    exit 1
fi

# Check for too many crops (max 2 crops = 4 GPUs)
if [ ${#CROPS[@]} -gt 2 ]; then
    echo "Error: Too many crops (${#CROPS[@]}). Maximum 2 crops allowed (4 GPUs available: 2 per crop)"
    exit 1
fi

# Load your environment
module load miniforge/24.3.0-0

echo "Starting best config tests: $MODEL / ${CROPS[*]} / $COUNTRY"
echo "Running overall and ahead_pred tests in parallel (2 GPUs per crop)"

mkdir -p data/best_config_tests log
rm -rf log/best_config*.log

run_test() {
    local gpu_id=$1
    local model=$2
    local crop=$3
    local country=$4
    local test_type=$5
    local log_file="log/best_config_${model}_${crop}_${country}_${test_type}.log"
    
    echo "Running: $model / $crop / $country / $test_type on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id TRANSFORMERS_NO_TORCHVISION=1 python -m src.crop_yield.best_config_tests \
        --model "$model" --crop-type "$crop" --country "$country" --test-type "$test_type" \
        --grid-search-results-dir data/results "${EXTRA_ARGS[@]}" \
        >> "$log_file" 2>&1
    
    echo "Completed: $model / $crop / $country / $test_type"
}

# Run tests for each crop (2 GPUs per crop: one for overall, one for ahead_pred)
gpu_counter=0
for crop in "${CROPS[@]}"; do
    # Run overall test on one GPU
    run_test $gpu_counter "$MODEL" "$crop" "$COUNTRY" "overall" &
    ((gpu_counter++))
    
    # Run ahead_pred test on another GPU
    run_test $gpu_counter "$MODEL" "$crop" "$COUNTRY" "ahead_pred" &
    ((gpu_counter++))
done

# Wait for all background jobs to complete
wait

echo "All best config tests completed!"
echo "Results saved to: data/best_config_tests/"
echo "Logs saved to: log/best_config_*.log"

# Summary of what was run
echo ""
echo "Summary:"
echo "Model: $MODEL"
echo "Crops: ${CROPS[*]}"
echo "Country: $COUNTRY"
echo "Total GPUs used: $gpu_counter"