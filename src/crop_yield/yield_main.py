import argparse
import logging
import os
import numpy as np
import torch
import random
from src.utils.utils import parse_args

# Set up deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.use_deterministic_algorithms(True)

from src.crop_yield.trainers.weatherbert_yield_trainer import (
    weatherbert_yield_training_loop,
)
from src.crop_yield.trainers.weatherformer_yield_trainer import (
    weatherformer_yield_training_loop,
)
from src.utils.constants import DATA_DIR
from src.utils.utils import setup_logging


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="model name weatherformer or weatherbert",
    default="weatherformer",
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=128, type=int)
parser.add_argument(
    "--n-past-years", help="number of past years to look at", default=6, type=int
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=20, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate for Adam", default=0.0005, type=float
)
parser.add_argument(
    "--decay_factor",
    help="learning rate exponential decay factor",
    default=0.98,
    type=float,
)
parser.add_argument(
    "--n-warmup-epochs", help="number of warmup epochs", default=5, type=int
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model weights",
    default="trained_models/weatherformer_60k_latest.pth",
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
    default=1.0,
    type=float,
)


def main():
    # Setup logging
    setup_logging(rank=0)  # Single GPU, rank always 0

    try:
        args_dict = parse_args(parser)

        # Determine which training function to use based on model type
        model_type = args_dict["model"].lower()

        if model_type == "weatherbert":
            avg_rmse = weatherbert_yield_training_loop(args_dict)
        elif model_type == "weatherformer":
            avg_rmse = weatherformer_yield_training_loop(args_dict)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Choose 'weatherbert' or 'weatherformer'"
            )

        logger = logging.getLogger(__name__)
        logger.info("Training completed successfully!")
        logger.info(f"Final average RMSE: {avg_rmse:.3f}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
