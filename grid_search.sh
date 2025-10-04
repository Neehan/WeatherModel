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

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            shift
            MODELS+=("$1")
            shift
            MODELS+=("$1")
            shift
            ;;
        --crop)
            shift
            CROPS+=("$1")
            shift
            CROPS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#MODELS[@]} -ne 2 ] || [ ${#CROPS[@]} -ne 2 ]; then
    echo "Usage: $0 --model <model1> <model2> --crop <crop1> <crop2> [additional_python_args...]"
    echo ""
    echo "Example: $0 --model weatherformer weatherformersinusoid --crop corn soybean"
    echo ""
    echo "Available models: weatherbert, weatherformer, decoder, weatherformersinusoid, decodersinusoid, weatherformermixture, weatherautoencodermixture, weatherautoencoder, weatherautoencodersinusoid, simmtm, cnnrnn, gnnrnn, linear, chronos"
    echo "Available crops: corn, soybean, wheat, sunflower, cotton, sugarcane, beans, corn_rainfed, beans_rainfed"
    exit 1
fi

valid_models=("weatherbert" "weatherformer" "decoder" "weatherformersinusoid" "decodersinusoid" "weatherformermixture" "weatherautoencodermixture" "weatherautoencoder" "weatherautoencodersinusoid" "simmtm" "cnnrnn" "gnnrnn" "linear" "chronos")
for model in "${MODELS[@]}"; do
    if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
        echo "Error: Invalid model '${model}'. Valid options: ${valid_models[@]}"
        exit 1
    fi
done

valid_crops=("corn" "soybean" "wheat" "sunflower" "cotton" "sugarcane" "beans" "corn_rainfed" "beans_rainfed")
for crop in "${CROPS[@]}"; do
    if [[ ! " ${valid_crops[@]} " =~ " ${crop} " ]]; then
        echo "Error: Invalid crop '${crop}'. Valid options: ${valid_crops[@]}"
        exit 1
    fi
done

module load miniforge/24.3.0-0

echo "Starting grid search: ${MODELS[*]} / ${CROPS[*]} (pretrained only)"

mkdir -p data/grid_search log
rm -rf data/trained_models/crop_yield/* log/gpu*.log

run_experiment() {
    CUDA_VISIBLE_DEVICES=$1 TRANSFORMERS_NO_TORCHVISION=1 python -m src.crop_yield.grid_search \
        --model "$2" --crop-type "$3" --load-pretrained --output-dir data/grid_search "${EXTRA_ARGS[@]}" \
        >> "log/gpu$1.log" 2>&1
}

run_experiment 0 "${MODELS[0]}" "${CROPS[0]}" &
run_experiment 1 "${MODELS[0]}" "${CROPS[1]}" &
run_experiment 2 "${MODELS[1]}" "${CROPS[0]}" &
run_experiment 3 "${MODELS[1]}" "${CROPS[1]}" &
wait 