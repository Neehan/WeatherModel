import argparse
import logging
import os
import numpy as np
import torch
import random
from typing import Optional
from src.utils.utils import get_model_params

# Set up deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.use_deterministic_algorithms(True)

from src.crop_yield.base.yield_dataloader import read_soybean_dataset
from src.crop_yield.yield_trainer import YieldTrainer
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
    "--lr_decay_factor",
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
    "--n-cross-validation-folds",
    help="number of cross validation folds",
    default=5,
    type=int,
)


def parse_args():
    args = parser.parse_args()
    args_dict = vars(args)

    logger = logging.getLogger(__name__)
    logger.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logger.info(f"{arg}: {value}")

    # Model size configuration
    model_size = args.model_size.lower()
    model_size_params = get_model_params(model_size)

    args_dict["model_size_params"] = model_size_params

    return args_dict


def main():
    # Setup logging
    setup_logging(rank=0)  # Single GPU, rank always 0

    try:
        args_dict = parse_args()

        # Load the soybean dataset
        soybean_df = read_soybean_dataset(DATA_DIR)

        # Run cross-validation training using the trainer
        avg_rmse = YieldTrainer.cross_validation_training_loop(soybean_df, args_dict)

        logger = logging.getLogger(__name__)
        logger.info("Training completed successfully!")
        logger.info(f"Final average RMSE: {avg_rmse:.3f}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
