import argparse
import logging
import os
import random

import numpy as np
import torch

from src.utils.utils import parse_args, setup_logging
from src.crop_yield.dataloader.cropnet_dataloader import read_cropnet_dataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="model name weatherformer, weatherformersinusoid, weatherformermixture, weatherbert, weatherautoencoder, weatherautoencodersine, or weathercnn",
    default="weatherformer",
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=64, type=int)
parser.add_argument(
    "--n-past-years", help="number of past years to look at", default=4, type=int
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
    default=0.95,
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
    default="mini",
    type=str,
)
parser.add_argument(
    "--n-train-years",
    help="number of years of training data to use (start year will be calculated as test_year - n_train_years + 1)",
    default=4,
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
    default=3,
    type=int,
)
parser.add_argument(
    "--crop-type",
    help="specific crop type to train (if not provided, trains all crops)",
    choices=["Cotton", "Corn", "Soybeans", "WinterWheat"],
    default=None,
    type=str,
)


def train_single_crop(crop_type: str, args_dict: dict):
    """Train model for a single crop type"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training for crop: {crop_type}")

    # Import training functions
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
    from src.crop_yield.trainers.weathercnn_yield_trainer import (
        weathercnn_yield_training_loop,
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
    from src.crop_yield.dataloader.cropnet_dataloader import (
        get_crop_rmse_conversion_factor,
    )

    # Create crop-specific args
    crop_args = args_dict.copy()
    crop_args["crop_type"] = crop_type

    # Determine which training function to use based on model type
    model_type = args_dict["model"].lower()

    if model_type == "weatherbert":
        cross_validation_results = weatherbert_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weatherformer":
        cross_validation_results = weatherformer_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weatherformersinusoid":
        cross_validation_results = weatherformer_sinusoid_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weatherformermixture":
        cross_validation_results = weatherformer_mixture_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weatherautoencodermixture":
        cross_validation_results = weatherautoencoder_mixture_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weatherautoencoder":
        cross_validation_results = weatherautoencoder_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weatherautoencodersine":
        cross_validation_results = weatherautoencoder_sine_yield_training_loop(
            crop_args, use_cropnet=True
        )
    elif model_type == "weathercnn":
        cross_validation_results = weathercnn_yield_training_loop(
            crop_args, use_cropnet=True
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'weatherbert', 'weatherformer', 'weatherformersinusoid', 'weatherformermixture', 'weatherautoencodermixture', 'weatherautoencoder', 'weatherautoencodersine', or 'weathercnn'"
        )

    # Convert MSE to RMSE using crop-specific scaling factor
    crop_std = get_crop_rmse_conversion_factor(crop_type)
    avg_best_rmse = cross_validation_results["avg_best_val_loss"] * crop_std
    std_best_rmse = cross_validation_results["std_best_val_loss"] * crop_std

    logger.info(f"Crop {crop_type} - Using scaling factor (std): {crop_std:.2f}")
    logger.info(
        f"Crop {crop_type} - Final RMSE: {avg_best_rmse:.3f} ± {std_best_rmse:.3f}"
    )

    logger.info(f"Completed training for crop: {crop_type}")
    return avg_best_rmse, std_best_rmse


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

    # Define the 4 crop types
    crop_types = ["Cotton", "Corn", "Soybeans", "WinterWheat"]
    # crop_types = ["Corn", "Soybeans", "WinterWheat"]

    logger = logging.getLogger(__name__)

    # Check if specific crop type was provided
    if args_dict.get("crop_type"):
        crop_types = [args_dict["crop_type"]]
        logger.info(
            f"Starting CropNet training for specific crop: {args_dict['crop_type']}"
        )
    else:
        logger.info("Starting CropNet training for all crops...")

    # Results storage
    all_results = {}

    # Train each crop separately
    for crop_type in crop_types:
        try:
            logger.info(f"=" * 50)
            logger.info(f"Training {crop_type}")
            logger.info(f"=" * 50)

            result = train_single_crop(crop_type, args_dict)
            all_results[crop_type] = result

            logger.info(f"Completed {crop_type} - Result: {result}")

        except Exception as e:
            logger.error(f"Failed to train {crop_type}: {e}")
            all_results[crop_type] = None

    # Print summary results
    logger.info("=" * 60)
    if args_dict.get("crop_type"):
        logger.info(f"CROPNET TRAINING SUMMARY - {args_dict['crop_type']}")
    else:
        logger.info("CROPNET TRAINING SUMMARY")
    logger.info("=" * 60)

    for crop_type, result in all_results.items():
        if result is not None:
            avg_rmse, std_rmse = result
            logger.info(f"{crop_type}: RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}")
        else:
            logger.info(f"{crop_type}: FAILED")

    if args_dict.get("crop_type"):
        logger.info(f"CropNet training completed for {args_dict['crop_type']}!")
    else:
        logger.info("CropNet training completed for all crops!")

    return all_results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"CropNet training failed with error: {e}")
        raise
