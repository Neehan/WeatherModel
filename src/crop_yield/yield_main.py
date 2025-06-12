import argparse
import logging
import os
import random

import numpy as np
import torch

from src.utils.utils import parse_args, setup_logging

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="model name weatherformer, weatherformermixture, weatherbert, weatherautoencoder, weatherautoencodersine, or weathercnn",
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
    "--cross-validation-k",
    help="number of cross validation folds",
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
    "--train-pct",
    help="percentage of training data to use (1-100)",
    default=100,
    type=int,
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
    default=7,
    type=int,
)


def main():
    # Setup logging
    setup_logging(rank=0)  # Single GPU, rank always 0

    try:
        args_dict = parse_args(parser)
        seed = args_dict["seed"]
        # Set up deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

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
        from src.crop_yield.trainers.weatherformer_yield_trainer import (
            weatherformer_yield_training_loop,
        )

        # Validate train-pct parameter
        if not 1 <= args_dict["train_pct"] <= 100:
            raise ValueError(
                f"train-pct must be between 1 and 100, got {args_dict['train_pct']}"
            )

        # Determine which training function to use based on model type
        model_type = args_dict["model"].lower()

        if model_type == "weatherbert":
            cross_validation_results = weatherbert_yield_training_loop(args_dict)
        elif model_type == "weatherformer":
            cross_validation_results = weatherformer_yield_training_loop(args_dict)
        elif model_type == "weatherformermixture":
            cross_validation_results = weatherformer_mixture_yield_training_loop(
                args_dict
            )
        elif model_type == "weatherautoencodermixture":
            cross_validation_results = weatherautoencoder_mixture_yield_training_loop(
                args_dict
            )
        elif model_type == "weatherautoencoder":
            cross_validation_results = weatherautoencoder_yield_training_loop(args_dict)
        elif model_type == "weatherautoencodersine":
            cross_validation_results = weatherautoencoder_sine_yield_training_loop(
                args_dict
            )
        elif model_type == "weathercnn":
            cross_validation_results = weathercnn_yield_training_loop(args_dict)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Choose 'weatherbert', 'weatherformer', 'weatherformermixture', 'weatherautoencodermixture', 'weatherautoencoder', 'weatherautoencodersine', or 'weathercnn'"
            )

        logger = logging.getLogger(__name__)
        logger.info("Training completed successfully!")

        # Convert MSE to RMSE for comparison with literature
        avg_best_rmse = (cross_validation_results["avg_best_val_loss"]) * 11.03
        std_best_rmse = cross_validation_results["std_best_val_loss"] * 11.03
        # 11.03 is the std of the dataset yield
        logger.info(
            f"Final average best RMSE: {avg_best_rmse:.3f} ± {std_best_rmse:.3f}"
        )

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
