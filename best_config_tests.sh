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
MODELS=()
CROPS=()
COUNTRY="usa"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --crops)
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
if [ ${#MODELS[@]} -eq 0 ] || [ ${#CROPS[@]} -eq 0 ]; then
    echo "Usage: $0 --models <model1> [model2] --crops <crop1> <crop2> [--country <country>] [additional_python_args...]"
    echo ""
    echo "Supported models: weatherformer, weatherautoencoder, weatherformersinusoid, simmtm, cnnrnn, linear, chronos, xgboost, randomforest"
    echo "GPU requirements: 1 model × 2 crops = 2 GPUs, or 2 models × 2 crops = 4 GPUs"
    echo ""
    echo "Examples:"
    echo "  $0 --models weatherformer --crops soybean corn                          # 2 GPUs"
    echo "  $0 --models weatherformer weatherautoencoder --crops soybean corn       # 4 GPUs"
    echo "  $0 --models weatherformer linear --crops soybean corn --country argentina"
    exit 1
fi

# Validate GPU requirements
total_tests=$((${#MODELS[@]} * ${#CROPS[@]}))

if [ ${#MODELS[@]} -lt 1 ] || [ ${#MODELS[@]} -gt 2 ]; then
    echo "Error: 1 or 2 models required (got ${#MODELS[@]})"
    exit 1
fi

if [ ${#CROPS[@]} -ne 2 ]; then
    echo "Error: Exactly 2 crops required (got ${#CROPS[@]})"
    exit 1
fi

if [ $total_tests -ne 2 ] && [ $total_tests -ne 4 ]; then
    echo "Error: Only 2 GPUs (1 model × 2 crops) or 4 GPUs (2 models × 2 crops) configurations supported"
    echo "Got: ${#MODELS[@]} models × ${#CROPS[@]} crops = $total_tests tests"
    exit 1
fi

# Load your environment
module load miniforge/24.3.0-0

echo "Starting best config tests: ${MODELS[*]} / ${CROPS[*]} / $COUNTRY"
echo "Running extreme year tests with weather cutoff at week 26"
echo "Total tests: ${#MODELS[@]} models × ${#CROPS[@]} crops = $((${#MODELS[@]} * ${#CROPS[@]})) tests"

# Determine log prefix based on number of GPUs
if [ $total_tests -eq 2 ]; then
    LOG_PREFIX="mit_gpu"
else
    LOG_PREFIX="gpu"
fi

mkdir -p data/best_config_tests log

run_test() {
    local gpu_id=$1
    local model=$2
    local crop=$3
    local country=$4
    local log_file="log/${LOG_PREFIX}${gpu_id}.log"
    
    echo "Running: $model / $crop on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id TRANSFORMERS_NO_TORCHVISION=1 python -m src.crop_yield.best_config_tests \
        --model "$model" --crop-type "$crop" --country "$country" \
        --grid-search-results-dir data/results "${EXTRA_ARGS[@]}" \
        >> "$log_file" 2>&1
    
    echo "Completed: $model / $crop on GPU $gpu_id"
}

# Run extreme year test for each model/crop combination (1 GPU per test)
gpu_counter=0
for model in "${MODELS[@]}"; do
    for crop in "${CROPS[@]}"; do
        # Run extreme year test with weather cutoff
        run_test $gpu_counter "$model" "$crop" "$COUNTRY" &
        ((gpu_counter++))
    done
done

# Wait for all background jobs to complete
wait

echo "All extreme year tests with weather cutoff completed!"
echo "Results saved to: data/best_config_tests/"
echo "Logs saved to: log/${LOG_PREFIX}*.log"

# Summary of what was run
echo ""
echo "Summary:"
echo "Models: ${MODELS[*]}"
echo "Crops: ${CROPS[*]}"
echo "Country: $COUNTRY"
echo "Test type: extreme year with weather cutoff at week 26"
echo "Total tests run: $gpu_counter"
echo "GPUs used: $gpu_counter"