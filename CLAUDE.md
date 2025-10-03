# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

WeatherModel is a comprehensive machine learning framework for agricultural weather modeling and crop yield prediction. The project implements transformer-based architectures including WeatherBERT and WeatherFormer for weather sequence modeling and crop yield forecasting.

## Architecture

### Core Components
- **Pretraining Models** (`src/pretraining/`): Self-supervised weather model pretraining
  - WeatherBERT: BERT-inspired transformer with masked weather feature prediction
  - WeatherFormer: Variational transformer with probabilistic outputs (μ, σ²)
  - WeatherAutoencoder: Autoencoder variants with mixture and sinusoid positional encodings
  - SimMTM: Similarity-based masked time modeling
  - MLP: Multi-layer perceptron baseline

- **Crop Yield Models** (`src/crop_yield/`): Supervised learning for yield prediction
  - Transfer learning from pretrained weather models
  - Support for corn, soybean, wheat, and other crops across multiple countries
  - Multiple baseline models (CNN-RNN, GNN-RNN, Linear, Chronos, XGBoost, RandomForest)
  - Hyperparameter tuning via `grid_search.py`
  - Best config evaluation via `best_config_tests.py`

- **Base Models** (`src/base_models/`): Shared model components
  - Transformer encoders with spatiotemporal and vanilla positional encodings
  - Weather CNN architectures
  - Soil CNN for incorporating soil data

### Data Pipeline
- **Weather Data**: NASA POWER daily/weekly/monthly weather datasets (31 meteorological features)
- **Crop Data**: County-level yield data across multiple countries (2000-2020)
  - USA: Corn Belt counties
  - International: Argentina, Brazil, Mexico
  - Crops: corn, soybean, wheat, sunflower, cotton, sugarcane, beans
- **Preprocessing**: Automated data downloaders and scalers in `src/weather_preprocessing/`
- **Data Storage**: PyTorch tensors for efficient loading (100+ files for weather data)

## Development Commands

### Environment Setup
```bash
# Automated setup
chmod +x installation.sh
./installation.sh

# Manual setup
conda create -n weather python=3.10 -y
conda activate weather
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Model Training

#### Weather Model Pretraining
```bash
# Single GPU
python -m src.pretraining.pretraining_main \
    --model weatherformer \
    --batch-size 64 \
    --n-epochs 30 \
    --init-lr 1e-6 \
    --model-size small

# Multi-GPU with SLURM (4 GPUs)
sbatch pretraining.sh weatherformer --batch-size 128 --model-size small

# Train multiple models sequentially
sbatch pretraining.sh weatherformer weatherbert --batch-size 128 --model-size small

# Resume from checkpoint
python -m src.pretraining.pretraining_main \
    --model weatherformer \
    --resume-from-checkpoint data/trained_models/pretraining/checkpoint.pth \
    --batch-size 64 \
    --n-epochs 30
```

#### Crop Yield Prediction
```bash
# Single model training
python -m src.crop_yield.yield_main \
    --model weatherformer \
    --pretrained-model-path data/trained_models/pretraining/weatherformer_mini.pth \
    --batch-size 128 \
    --n-past-years 6 \
    --cross-validation-k 5

# Without pretraining
python -m src.crop_yield.yield_main \
    --model weatherformer \
    --batch-size 128 \
    --n-past-years 6 \
    --cross-validation-k 5

# SLURM cluster
sbatch crop_yield.sh --model weatherbert --n-past-years 8
```

#### Grid Search (Hyperparameter Tuning)
```bash
# Single model: pretrained vs not pretrained (2 GPUs)
sbatch grid_search.sh --model weatherformer --crop corn

# Two models: each with pretrained vs not (4 GPUs)
sbatch grid_search.sh --model weatherformer weatherformersinusoid --crop corn

# One model, two crops (4 GPUs)
sbatch grid_search.sh --model weatherformer --crop corn soybean

# Baseline models (no pretrained variants)
sbatch grid_search.sh --model xgboost randomforest --crop soybean wheat

# With country specification
sbatch grid_search.sh --model weatherformer --crop corn --country argentina
```

#### Best Config Tests (Extreme Year Evaluation)
```bash
# Run best configurations on extreme year test
# Requires 2 models × 2 crops = 4 GPUs
sbatch best_config_tests.sh --models weatherformer weatherautoencoder --crops soybean corn

# With country specification
sbatch best_config_tests.sh --models weatherformer linear --crops soybean wheat --country argentina
```

### Model Sizes and Parameters
- **mini**: ~60K parameters (fast prototyping)
- **small**: ~2M parameters (good balance)
- **medium**: ~8M parameters (higher capacity)
- **large**: ~56M parameters (maximum performance)

### Key Arguments

#### Pretraining Arguments
- `--model`: Model type (weatherbert, weatherformer, weatherformersinusoid, weatherformermixture, weatherautoencoder, simmtm, mlp)
- `--model-size`: Model capacity (mini, small, medium, large)
- `--batch-size`: Training batch size (higher for multi-GPU)
- `--n-epochs`: Number of training epochs
- `--init-lr`: Initial learning rate
- `--beta`: Variational loss weight (WeatherFormer models only)
- `--masking-prob`: Feature masking probability (WeatherBERT only)
- `--n-masked-features`: Number of features to predict (WeatherFormer models)
- `--resume-from-checkpoint`: Path to resume training from checkpoint

#### Crop Yield Arguments
- `--model`: Model type (weatherbert, weatherformer, weatherformersinusoid, weatherformermixture, decoder, decodersinusoid, weatherautoencoder, weatherautoencodersine, simmtm, cnnrnn, gnnrnn, linear, chronos, xgboost, randomforest)
- `--pretrained-model-path`: Path to pretrained weights for transfer learning (if omitted, trains from scratch)
- `--crop-type`: Crop to predict (corn, soybean, wheat, sunflower, cotton, sugarcane, beans)
- `--country`: Country dataset (usa, argentina, brazil, mexico; default: usa)
- `--n-past-years`: Historical years of weather data (4-10 recommended)
- `--cross-validation-k`: Number of CV folds for robust evaluation
- `--batch-size`: Training batch size
- `--n-epochs`: Number of training epochs
- `--init-lr`: Initial learning rate
- `--decay-factor`: Learning rate exponential decay factor
- `--n-warmup-epochs`: Number of warmup epochs
- `--beta`: Uncertainty regularization (WeatherFormer yield models)

## Data Organization

### Input Data Structure
- `data/nasa_power/`: Weather datasets (PyTorch tensors)
  - `pytorch/`: Daily weather data
  - `processed/`: Weekly aggregated data with scalers
- `data/khaki_soybeans/`: Crop yield datasets
- `data/USDA/`: Government yield statistics

### Model Outputs
- `data/trained_models/pretraining/`: Pretrained weather models
- `data/trained_models/crop_yield/`: Fine-tuned yield prediction models
- `data/grid_search/`: Hyperparameter search results (.tsv files)

## Testing and Validation

### Cross-Validation
The framework uses K-fold cross-validation across counties for robust evaluation. Use `--cross-validation-k 5` for standard 5-fold CV.

### Evaluation Metrics
- **Primary**: Root Mean Square Error (RMSE) in bushels/acre
- **Secondary**: R² correlation, extreme year performance (when yield deviates >1σ from 5-year mean)

### Baseline Comparisons
- CNN-LSTM approaches
- Linear regression baselines  
- Non-pretrained vs pretrained model variants

## Multi-GPU and Distributed Training

The repository supports distributed training using PyTorch DDP:
- **Pretraining**: Uses `torchrun` with 4 GPUs via [pretraining.sh](pretraining.sh)
- **Crop Yield**: Single GPU via [crop_yield.sh](crop_yield.sh)
- **Grid Search**: Parallel experiments on 4 GPUs via [grid_search.sh](grid_search.sh)
  - Each GPU runs an independent experiment (model/crop/pretrained combination)
  - Non-baseline models: 2 experiments per model (pretrained vs not pretrained)
  - Baseline models (xgboost, randomforest): 1 experiment per model
- **Best Config Tests**: Parallel evaluation on 4 GPUs via [best_config_tests.sh](best_config_tests.sh)
  - Each GPU evaluates one model/crop combination using best hyperparameters
  - Runs extreme year tests with weather cutoff at week 26

### SLURM Configuration
All SLURM scripts are configured for MIT Supercloud with:
- Partition: `mit_preemptable`
- Account: `mit_general`
- GPU types: H100 (pretraining), L40S (grid search/tests)
- Environment: `miniforge/24.3.0-0`

## File Naming Conventions

### Model Checkpoints
Format: `{model}_{task}_{param_count}_{status}.pth`
- Example: `weatherformer_yield_134.3k_best.pth`
- Includes automatic parameter counting and best/latest checkpoints

### Results Files  
- Grid search: `grid_search_{model}_{pretrained/not_pretrained}_{crop}.tsv`
- Training outputs: `{model}_{param_count}_output.json`

## Important Implementation Notes

### Memory Management
- Weather datasets are large (100+ PyTorch tensor files)
- Use distributed loading and caching for efficient training
- Weekly aggregated data recommended for most experiments

### Reproducibility
- All experiments use fixed random seeds
- Cross-validation splits are deterministic by county ID
- Model configurations saved in output JSON files

### Model Name Mappings (Paper Submission)
For anonymous submission, the following model name mappings are used:
- **T-BERT** → `weatherautoencoder`
- **VITA with std normal prior** → `weatherformer`
- **VITA with sinusoidal prior** → `weatherformersinusoid`
- **SimMTM** → `simmtm`
- **25-6 feature prediction experiment** → `mlp`

### Grid Search GPU Allocation Rules
- **Single model + single crop**: 2 GPUs (pretrained vs not pretrained)
- **Two models + single crop**: 4 GPUs (2 models × pretrained/not)
- **Single model + two crops**: 4 GPUs (2 crops × pretrained/not)
- **Baseline models**: No pretrained variants, 1 GPU per model/crop combination
- **Unsupported**: 2 non-baseline models + 2 crops (would need 8 GPUs)

### Checkpoint Resumption
All training scripts support `--resume-from-checkpoint` for fault tolerance during long training runs.