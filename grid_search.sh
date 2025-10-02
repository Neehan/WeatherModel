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

# Initialize variables
MODELS=()
CROPS=()
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --crop)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                CROPS+=("$1")
                shift
            done
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ ${#MODELS[@]} -eq 0 ] || [ ${#CROPS[@]} -eq 0 ]; then
    echo "Usage: $0 --model <model1> [model2] --crop <crop1> [crop2] [additional_python_args...]"
    echo ""
    echo "Supported configurations:"
    echo "  --model <model> --crop <crop>                    # Single model, single crop (2 GPUs)"
    echo "  --model <model1> <model2> --crop <crop>          # Two models, single crop (4 GPUs)"
    echo "  --model <model> --crop <crop1> <crop2>           # Single model, two crops (4 GPUs)"
    echo ""
    echo "Examples:"
    echo "  $0 --model weatherformer --crop corn"
    echo "  $0 --model weatherformer --crop corn --country argentina"
    echo "  $0 --model weatherformer --crop wheat --country mexico"
    echo "  $0 --model weatherformer weatherformersinusoid --crop corn"
    echo "  $0 --model weatherformer --crop corn soybean"
    echo ""
    echo "Available models: weatherbert, weatherformer, decoder, weatherformersinusoid, decodersinusoid, weatherformermixture, weatherautoencodermixture, weatherautoencoder, weatherautoencodersinusoid, simmtm, cnnrnn, gnnrnn, linear, chronos"
    echo "Available crops: corn, soybean, wheat, sunflower, cotton, sugarcane, beans, corn_rainfed, beans_rainfed"
    echo "Available countries: usa, argentina, brazil, mexico (default: usa)"
    exit 1
fi

# Check for unsupported case: 2 models + 2 crops = 8 experiments (we only have 4 GPUs)
if [ ${#MODELS[@]} -eq 2 ] && [ ${#CROPS[@]} -eq 2 ]; then
    echo "Error: Cannot run 2 models with 2 crops (would require 8 GPUs, but only 4 available)"
    echo "Supported configurations:"
    echo "  - 1 model + 1 crop (2 experiments: pretrained vs not pretrained)"
    echo "  - 2 models + 1 crop (4 experiments: 2 models × pretrained/not pretrained)"
    echo "  - 1 model + 2 crops (4 experiments: 2 crops × pretrained/not pretrained)"
    exit 1
fi


# Validate model names
valid_models=("weatherbert" "weatherformer" "decoder" "weatherformersinusoid" "decodersinusoid" "weatherformermixture" "weatherautoencodermixture" "weatherautoencoder" "weatherautoencodersinusoid" "simmtm" "cnnrnn" "gnnrnn" "linear" "chronos")
for model in "${MODELS[@]}"; do
    if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
        echo "Error: Invalid model '${model}'. Valid options: ${valid_models[@]}"
        exit 1
    fi
done

# Validate crop names
valid_crops=("corn" "soybean" "wheat" "sunflower" "cotton" "sugarcane" "beans" "corn_rainfed" "beans_rainfed")
for crop in "${CROPS[@]}"; do
    if [[ ! " ${valid_crops[@]} " =~ " ${crop} " ]]; then
        echo "Error: Invalid crop '${crop}'. Valid options: ${valid_crops[@]}"
        exit 1
    fi
done

# Load your environment
module load miniforge/24.3.0-0

echo "Starting grid search: ${MODELS[*]} / ${CROPS[*]}"

mkdir -p data/grid_search log
rm -rf data/trained_models/crop_yield/* log/gpu*.log

run_experiment() {
    CUDA_VISIBLE_DEVICES=$1 TRANSFORMERS_NO_TORCHVISION=1 python -m src.crop_yield.grid_search \
        --model "$2" --crop-type "$3" $4 --output-dir data/grid_search "${EXTRA_ARGS[@]}" \
        >> "log/gpu$1.log" 2>&1
}

if [ ${#MODELS[@]} -eq 1 ] && [ ${#CROPS[@]} -eq 1 ]; then
    run_experiment 0 "${MODELS[0]}" "${CROPS[0]}" "" &
    run_experiment 1 "${MODELS[0]}" "${CROPS[0]}" "--load-pretrained" &
    wait
elif [ ${#MODELS[@]} -eq 2 ] && [ ${#CROPS[@]} -eq 1 ]; then
    run_experiment 0 "${MODELS[0]}" "${CROPS[0]}" "" &
    run_experiment 1 "${MODELS[0]}" "${CROPS[0]}" "--load-pretrained" &
    run_experiment 2 "${MODELS[1]}" "${CROPS[0]}" "" &
    run_experiment 3 "${MODELS[1]}" "${CROPS[0]}" "--load-pretrained" &
    wait
elif [ ${#MODELS[@]} -eq 1 ] && [ ${#CROPS[@]} -eq 2 ]; then
    run_experiment 0 "${MODELS[0]}" "${CROPS[0]}" "" &
    run_experiment 1 "${MODELS[0]}" "${CROPS[0]}" "--load-pretrained" &
    run_experiment 2 "${MODELS[0]}" "${CROPS[1]}" "" &
    run_experiment 3 "${MODELS[0]}" "${CROPS[1]}" "--load-pretrained" &
    wait
fi 