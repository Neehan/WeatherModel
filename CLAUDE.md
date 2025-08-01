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
  - Support for corn, soybean, and winter wheat prediction
  - Multiple baseline models (CNN-RNN, GNN-RNN, Linear)

- **Base Models** (`src/base_models/`): Shared model components
  - Transformer encoders with spatiotemporal and vanilla positional encodings
  - Weather CNN architectures
  - Soil CNN for incorporating soil data

### Data Pipeline
- **Weather Data**: NASA POWER daily/weekly/monthly weather datasets (31 meteorological features)
- **Crop Data**: County-level yield data from US Corn Belt (2000-2020)
- **Preprocessing**: Automated data downloaders and scalers in `src/weather_preprocessing/`

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

# Multi-GPU with SLURM
chmod +x pretraining.sh
sbatch pretraining.sh --model weatherformer --batch-size 128 --model-size small
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

# SLURM cluster
chmod +x crop_yield.sh
sbatch crop_yield.sh --model weatherbert --n-past-years 8
```

#### Grid Search
```bash
# Single model comparison (pretrained vs not pretrained)
chmod +x grid_search.sh
sbatch grid_search.sh weatherformer corn

# Two model comparison  
sbatch grid_search.sh weatherformer weatherformersinusoid soybean
```

### Model Sizes and Parameters
- **mini**: ~60K parameters (fast prototyping)
- **small**: ~2M parameters (good balance)
- **medium**: ~8M parameters (higher capacity)
- **large**: ~56M parameters (maximum performance)

### Key Arguments
- `--model`: Model type (weatherbert, weatherformer, weatherformersinusoid, weatherformermixture, etc.)
- `--pretrained-model-path`: Path to pretrained weights for transfer learning
- `--n-past-years`: Historical years of weather data (4-10 recommended)
- `--cross-validation-k`: Number of CV folds for robust evaluation
- `--beta`: Variational loss weight for WeatherFormer models
- `--masking-prob`: Feature masking probability for WeatherBERT
- `--n-masked-features`: Number of features to predict for WeatherFormer

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

## Multi-GPU Training

The repository supports distributed training using PyTorch DDP:
- Use `torchrun` for multi-GPU on single node
- SLURM scripts automatically handle multi-node distributed training
- All training scripts support `--resume-from-checkpoint` for fault tolerance

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

### SLURM Integration
- All shell scripts configured for MIT Supercloud SLURM
- Automatic GPU allocation and environment loading
- Progress monitoring with real-time log files