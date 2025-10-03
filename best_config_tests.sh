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
    echo "Usage: $0 --models <model1> <model2> --crops <crop1> <crop2> [--country <country>] [additional_python_args...]"
    echo ""
    echo "Supported models: weatherformer, weatherautoencoder, weatherformersinusoid, simmtm, cnnrnn, linear, chronos, xgboost, randomforest"
    echo ""
    echo "Examples:"
    echo "  $0 --models weatherformer weatherautoencoder --crops soybean corn"
    echo "  $0 --models weatherformer linear --crops soybean corn --country argentina"
    echo "  $0 --models xgboost randomforest --crops soybean wheat"
    exit 1
fi

# Check for exactly 2 models and 2 crops (4 total tests = 4 GPUs)
if [ ${#MODELS[@]} -ne 2 ]; then
    echo "Error: Exactly 2 models required (got ${#MODELS[@]})"
    exit 1
fi

if [ ${#CROPS[@]} -ne 2 ]; then
    echo "Error: Exactly 2 crops required (got ${#CROPS[@]})"
    exit 1
fi

# Load your environment
module load miniforge/24.3.0-0

echo "Starting best config tests: ${MODELS[*]} / ${CROPS[*]} / $COUNTRY"
echo "Running extreme year tests with weather cutoff at week 26"
echo "Total tests: ${#MODELS[@]} models × ${#CROPS[@]} crops = $((${#MODELS[@]} * ${#CROPS[@]})) tests"

mkdir -p data/best_config_tests log
rm -rf log/best_config*.log

run_test() {
    local gpu_id=$1
    local model=$2
    local crop=$3
    local country=$4
    local log_file="log/best_config_${model}_${crop}_${country}_extreme_weather_cutoff.log"
    
    echo "Running: $model / $crop / $country / extreme_weather_cutoff on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id TRANSFORMERS_NO_TORCHVISION=1 python -m src.crop_yield.best_config_tests \
        --model "$model" --crop-type "$crop" --country "$country" \
        --grid-search-results-dir data/results "${EXTRA_ARGS[@]}" \
        >> "$log_file" 2>&1
    
    echo "Completed: $model / $crop / $country / extreme_weather_cutoff"
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
echo "Logs saved to: log/best_config_*.log"

# Summary of what was run
echo ""
echo "Summary:"
echo "Models: ${MODELS[*]}"
echo "Crops: ${CROPS[*]}"
echo "Country: $COUNTRY"
echo "Test type: extreme year with weather cutoff at week 26"
echo "Total tests run: $gpu_counter"
echo "GPUs used: $gpu_counter"