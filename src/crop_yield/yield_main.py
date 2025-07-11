import argparse
import logging
import os
import random

import numpy as np
import torch

from src.utils.utils import parse_args, setup_logging
from src.utils.constants import CROP_YIELD_STATS

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="model name weatherformer, weatherformersinusoid, weatherformermixture, weatherbert, weatherautoencoder, weatherautoencodersine, cnnrnn, or linear",
    default="weatherformer",
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=64, type=int)
parser.add_argument(
    "--n-past-years", help="number of past years to look at", default=6, type=int
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=40, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate for Adam", default=0.0005, type=float
)
parser.add_argument(
    "--decay_factor",
    help="learning rate exponential decay factor",
    default=None,
    type=float,
)
parser.add_argument(
    "--n-warmup-epochs", help="number of warmup epochs", default=10, type=int
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model weights",
    default=None,
    type=str,
)
parser.add_argument(
    "--model-size",
    help="model size mini (60k) small (2M), medium (8M), and large (56M)",
    default="small",
    type=str,
)
parser.add_argument(
    "--n-train-years",
    help="number of years of training data to use (start year will be calculated as test_year - n_train_years + 1)",
    default=5,
    type=int,
)
parser.add_argument(
    "--beta",
    help="beta parameter for WeatherFormer variational loss (sigma_y_squared)",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--use-optimal-lr",
    help="whether to find and use optimal learning rate before training",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--seed",
    help="seed for random number generator",
    default=1234,
    type=int,
)
parser.add_argument(
    "--n-mixture-components",
    help="number of gaussian mixture components for WeatherFormerMixture model",
    default=1,
    type=int,
)
parser.add_argument(
    "--crop-type",
    help="crop type to predict: soybean, corn or winter_wheat",
    default="soybean",
    type=str,
    choices=["soybean", "corn", "winter_wheat"],
)
parser.add_argument(
    "--test-year",
    help="specific test year for single-year evaluation (if not provided, uses 5-fold cross validation)",
    default=None,
    type=int,
)


def main(args_dict=None):
    # Setup logging
    setup_logging(rank=0)  # Single GPU, rank always 0

    if args_dict is None:
        args_dict = parse_args(parser)

    seed = args_dict["seed"]
    # Set up deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if args_dict["n_train_years"] < args_dict["n_past_years"] + 1:
        logging.warning(
            f"Not enough training data for current year + n_past_years. Required: {args_dict['n_past_years'] + 1}. "
            f"Available training years: {args_dict['n_train_years']}. "
            f"Setting n_past_years to {args_dict['n_train_years'] - 1}."
        )
        args_dict["n_past_years"] = args_dict["n_train_years"] - 1

    from src.crop_yield.trainers.weatherautoencoder_mixture_yield_trainer import (
        weatherautoencoder_mixture_yield_training_loop,
    )
    from src.crop_yield.trainers.weatherautoencoder_sine_yield_trainer import (
        weatherautoencoder_sine_yield_training_loop,
    )
    from src.crop_yield.trainers.weatherautoencoder_yield_trainer import (
        weatherautoencoder_yield_training_loop,
    )
    from src.crop_yield.trainers.weatherbert_yield_trainer import (
        weatherbert_yield_training_loop,
    )
    from src.crop_yield.trainers.cnnrnn_yield_trainer import (
        cnnrnn_yield_training_loop,
    )
    from src.crop_yield.trainers.weatherformer_mixture_yield_trainer import (
        weatherformer_mixture_yield_training_loop,
    )
    from src.crop_yield.trainers.weatherformer_sinusoid_yield_trainer import (
        weatherformer_sinusoid_yield_training_loop,
    )
    from src.crop_yield.trainers.weatherformer_yield_trainer import (
        weatherformer_yield_training_loop,
    )
    from src.crop_yield.trainers.linear_yield_trainer import (
        linear_yield_training_loop,
    )

    # Determine which training function to use based on model type
    model_type = args_dict["model"].lower()

    if model_type == "weatherbert":
        cross_validation_results = weatherbert_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "weatherformer":
        cross_validation_results = weatherformer_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "weatherformersinusoid":
        cross_validation_results = weatherformer_sinusoid_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "weatherformermixture":
        cross_validation_results = weatherformer_mixture_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "weatherautoencodermixture":
        cross_validation_results = weatherautoencoder_mixture_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "weatherautoencoder":
        cross_validation_results = weatherautoencoder_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "weatherautoencodersine":
        cross_validation_results = weatherautoencoder_sine_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "cnnrnn":
        cross_validation_results = cnnrnn_yield_training_loop(
            args_dict, use_cropnet=False
        )
    elif model_type == "linear":
        cross_validation_results = linear_yield_training_loop(
            args_dict, use_cropnet=False
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'weatherbert', 'weatherformer', 'weatherformersinusoid', 'weatherformermixture', 'weatherautoencodermixture', 'weatherautoencoder', 'weatherautoencodersine', 'cnnrnn', or 'linear'"
        )

    logger = logging.getLogger(__name__)
    logger.info("Training completed successfully!")

    kfold_results = cross_validation_results["fold_results"]
    fold_stds = CROP_YIELD_STATS[args_dict["crop_type"]]["std"]

    # Compute RMSE in bu/acre
    best_rmse_bu_acre = [result * std for result, std in zip(kfold_results, fold_stds)]
    avg_best_rmse = float(np.mean(best_rmse_bu_acre))
    std_best_rmse = float(np.std(best_rmse_bu_acre))

    # Compute R² for each fold: R² = 1 - (RMSE/std)²
    r_squared_values = [
        1 - (rmse / std) ** 2 for rmse, std in zip(best_rmse_bu_acre, fold_stds)
    ]
    avg_r_squared = float(np.mean(r_squared_values))
    std_r_squared = float(np.std(r_squared_values))

    logger.info(
        f"Final average best RMSE for {args_dict['crop_type']}: {avg_best_rmse:.3f} ± {std_best_rmse:.3f}"
    )
    logger.info(
        f"Final average R² for {args_dict['crop_type']}: {avg_r_squared:.3f} ± {std_r_squared:.3f}"
    )

    # Return individual R² values in addition to aggregated statistics
    return avg_best_rmse, std_best_rmse, avg_r_squared, std_r_squared, r_squared_values


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        raise
