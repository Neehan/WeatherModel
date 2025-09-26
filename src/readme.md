# VITA: Variational Inference Transformer for Asymmetric Data

## Model Name Mappings

For the anonymous submission, the following model name mappings are used:

- **T-BERT** → `weatherautoencoder`
- **VITA with std normal prior** → `weatherformer`  
- **VITA with sinusoidal prior** → `weatherformersinusoid`
- **SimMTM** → `simmtm`
- **25-6 feature prediction experiment** → `mlp`

## Code Structure

### Core Components

- **`base_models/`** - Contains main implementations for all model architectures
- **`base_trainer/`** - Contains main implementations for trainer classes and training utilities

### Pretraining

- **`pretraining/`** - Self-supervised pretraining framework
  - `models/` - Model implementations for pretraining tasks
  - `trainers/` - Training logic for pretraining models
  - `dataloader/` - Data loading utilities for pretraining
  - `pretraining_main.py` - Main entry point for pretraining experiments

### Crop Yield Prediction

- **`crop_yield/`** - Downstream crop yield prediction task
  - `models/` - Crop yield prediction model implementations
  - `trainers/` - Training logic for crop yield models
  - `dataloader/` - Data loading utilities for crop yield task
  - `grid_search.py` - **Main script for hyperparameter grid search**
  - `yield_main.py` - Main entry point for crop yield experiments

### Data Processing

- **`weather_preprocessing/`** - Weather data preprocessing utilities
  - `nasa_power/` - NASA POWER dataset processing
  - `noaa/` - NOAA dataset processing (not used)

### Utilities

- **`utils/`** - Common utilities and helper functions
  - `constants.py` - Global constants and configuration
  - `losses.py` - Loss function implementations
  - `utils.py` - General utility functions

## Experiment Execution

### Grid Search
For crop yield experiments, hyperparameter tuning is performed using:
```bash
python src/crop_yield/grid_search.py
```

### Model Training
Individual model training can be executed through the respective main scripts in each task directory.

## Anonymous Submission Notes

- All model implementations follow consistent interfaces defined in `base_models/`
- Training procedures follow consistent patterns defined in `base_trainer/`
- Hyperparameter optimization is centralized through grid search functionality
- Data preprocessing is modular and task-agnostic where possible
