# WeatherModel

Advanced weather modeling and crop yield prediction using transformer-based architectures.

## Overview

WeatherModel is a comprehensive machine learning framework designed for agricultural applications, featuring state-of-the-art transformer models for weather analysis and crop yield prediction. The project implements two main components:

1. **Weather Model Pretraining**: Self-supervised learning on large-scale weather datasets
2. **Crop Yield Prediction**: Supervised learning for soybean yield forecasting

## Key Features

- **WeatherBERT**: BERT-inspired transformer for weather sequence modeling with masked feature prediction
- **WeatherFormer**: Variational transformer architecture with probabilistic outputs (μ, σ²) for uncertainty quantification
- **Multi-scale temporal modeling**: Support for sequences up to 365 days with flexible granularity
- **Distributed training**: Multi-GPU support using PyTorch DDP
- **Cross-validation**: K-fold validation for robust model evaluation
- **Pretrained models**: Transfer learning from weather pretraining to downstream tasks

## Model Architectures

### WeatherBERT
- Transformer encoder architecture with masked weather feature prediction
- Input: Weather sequences (31 features) + coordinates + temporal information
- Training objective: Reconstruct masked weather features (15% masking probability)
- Supports multiple model sizes: mini (60K), small (2M), medium (8M), large (56M) parameters

### WeatherFormer  
- Extends WeatherBERT with variational outputs for uncertainty estimation
- Outputs probabilistic predictions (mean and variance)
- VAE-style parameterization with β-weighted KL divergence loss
- Ideal for applications requiring uncertainty quantification

## Installation

### Quick Setup
```bash
chmod +x installation.sh
./installation.sh
```

### Manual Setup
1. Install Miniconda/Anaconda
2. Create environment:
```bash
conda create -n weather python=3.10 -y
conda activate weather
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

4. Download datasets:
```bash
python data_downloader.py
```

## Usage

### Weather Model Pretraining

Train weather models on large-scale NASA POWER daily weather dataset for self-supervised representation learning.

#### Single GPU Training
```bash
python -m src.pretraining.pretraining_main \
    --model weatherbert \
    --batch-size 64 \
    --n-epochs 30 \
    --init-lr 1e-6 \
    --model-size mini
```

#### Multi-GPU Training (Recommended)
```bash
chmod +x pretraining.sh
sbatch pretraining.sh --model weatherformer --batch-size 128 --model-size small
```

#### Key Parameters
- `--model`: Choose `weatherbert` or `weatherformer`
- `--model-size`: Model capacity (`mini`, `small`, `medium`, `large`)
- `--masking-prob`: Fraction of features to mask (WeatherBERT only)
- `--n-masked-features`: Number of features to predict (WeatherFormer only)
- `--beta`: Variational loss weight (WeatherFormer only)

### Crop Yield Prediction

Fine-tune pretrained weather models on soybean yield data from the US Corn Belt.

#### Training
```bash
python -m src.crop_yield.yield_main \
    --model weatherformer \
    --pretrained-model-path data/trained_models/pretraining/weatherformer_mini.pth \
    --batch-size 128 \
    --n-past-years 6 \
    --cross-validation-k 5
```

#### SLURM Cluster
```bash
chmod +x crop_yield.sh
sbatch crop_yield.sh --model weatherbert --n-past-years 8
```

#### Key Parameters
- `--pretrained-model-path`: Path to pretrained weather model
- `--n-past-years`: Historical years of weather data to use (4-10 recommended)
- `--cross-validation-k`: Number of CV folds for robust evaluation
- `--beta`: Uncertainty regularization (WeatherFormer only)

## Data

### Weather Data
- **Source**: NASA POWER Daily Weather dataset
- **Coverage**: Global daily weather observations
- **Features**: 31 meteorological variables (temperature, precipitation, radiation, etc.)
- **Format**: PyTorch tensors for efficient loading
- **Access**: Automatically downloaded via HuggingFace Hub

### Crop Yield Data
- **Source**: US Corn Belt soybean yield records
- **Coverage**: 9 states with historical yield data (2000-2020)
- **Features**: County-level yields + weather + soil properties
- **Format**: CSV with spatial coordinates and temporal indexing

## Model Performance

### Pretraining Objectives
- **WeatherBERT**: Masked weather feature reconstruction (MAE loss)
- **WeatherFormer**: Probabilistic feature prediction with uncertainty (Variational loss)

### Crop Yield Metrics
- **Primary**: Root Mean Square Error (RMSE) in bushels/acre
- **Evaluation**: 5-fold cross-validation across counties
- **Baseline**: State-of-the-art CNN-LSTM approaches


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.
