#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=crop_grid
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=1                      # One task (master launcher)
#SBATCH --cpus-per-task=4              # More CPUs for parallel processing
#SBATCH --gres=gpu:h100:2              # Request 2 GPUs instead of 4
#SBATCH --mem=64GB                     # More memory per GPU (32GB per GPU)
#SBATCH -t 24:00:00                    # 24-hour wall time (grid search will take long)

# Load your environment
module load miniforge/24.3.0-0
source activate torch  # Replace with your conda env

echo "======== Starting Crop Yield Grid Search ========"
echo "Using 2 GPUs for parallel grid search"
echo "Total experiments: $(python -c "
beta_values = [0.0, 1e-4, 1e-3, 1e-2]
n_train_years = [5, 10, 20, 30] 
models = 2  # weatherformersinusoid, weatherformermixture
pretrained = 2  # True, False
print(len(beta_values) * len(n_train_years) * models * pretrained)
")"

python -m src.crop_yield.grid_search

echo "======== Grid Search Completed =========" 