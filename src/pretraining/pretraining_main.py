import argparse
import logging
import os
import torch
import torch.distributed as dist
from src.pretraining.trainers.weatherformer_trainer import weatherformer_training_loop
from src.pretraining.trainers.weatherbert_trainer import weatherbert_training_loop
from src.pretraining.trainers.weatherautoencoder_trainer import (
    weatherautoencoder_training_loop,
)
from src.pretraining.trainers.weatherformer_mixture_trainer import (
    weatherformer_mixture_training_loop,
)
from src.utils.utils import setup_distributed, cleanup_distributed, setup_logging
from src.utils.utils import parse_args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="model type is weatherformer, weatherbert, weatherautoencoder, or weatherformermixture",
    default="weatherformer",
    type=str,
)
parser.add_argument(
    "--resume-from-checkpoint",
    help="path to resume from checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model to load before training",
    default=None,
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=512, type=int)
parser.add_argument(
    "--n-masked-features",
    help="number of masked features (for weatherformer) the rest of the features are input features",
    default=10,
    type=int,
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=20, type=int
)
parser.add_argument("--init-lr", help="initial learning rate", default=1e-4, type=float)
parser.add_argument(
    "--n-warmup-epochs", help="number of warm-up epochs", default=10, type=float
)
parser.add_argument(
    "--decay-factor",
    help="exponential learning rate decay factor after warmup",
    default=0.99,
    type=float,
)
parser.add_argument(
    "--model-size",
    help="model size mini (60k), small (2M), medium (8M), and large (56M)",
    default="small",
    type=str,
)
parser.add_argument(
    "--masking-prob",
    help="percent to mask (for weatherbert)",
    default=0.15,
    type=float,
)
parser.add_argument(
    "--n-mixture-components",
    help="number of mixture components (for weatherformermixture)",
    default=7,
    type=int,
)
parser.add_argument(
    "--prior-weight",
    help="lambda parameter for mixture prior loss weighting (for weatherformermixture)",
    default=0.001,
    type=float,
)


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging
    setup_logging(rank)

    try:
        args_dict = parse_args(parser)

        # Add distributed training info to args
        args_dict["rank"] = rank
        args_dict["world_size"] = world_size
        args_dict["local_rank"] = local_rank

        model_type = args_dict["model"].lower()

        if model_type == "weatherformer":
            weatherformer_training_loop(args_dict)
        elif model_type == "weatherbert":
            weatherbert_training_loop(args_dict)
        elif model_type == "weatherautoencoder":
            weatherautoencoder_training_loop(args_dict)
        elif model_type == "weatherformermixture":
            weatherformer_mixture_training_loop(args_dict)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Choose 'weatherformer', 'weatherbert', 'weatherautoencoder', or 'weatherformermixture'"
            )
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
