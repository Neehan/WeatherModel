import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)


import numpy as np
import argparse

from .train import training_loop
from .model import Weatherformer
from .constants import *

np.random.seed(1234)
torch.manual_seed(1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument(
        "--n_input_features", help="number of input features", default=26, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of training epochs", default=75, type=int
    )
    parser.add_argument(
        "--init_lr", help="initial learning rate", default=0.0005, type=float
    )
    parser.add_argument(
        "--n_warmup_epochs", help="number of warm-up epochs", default=5, type=float
    )
    parser.add_argument(
        "--decay_factor",
        help="exponential learning rate decay factor after warmup",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--model_size",
        help="model size small (2M), medium (8M), and large (25M)",
        default="small",
        type=str,
    )

    args = parser.parse_args()
    args_dict = vars(args)
    logging.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logging.info(f"{arg}: {value}")

    model_size = args.model_size.lower()
    if model_size == "small":
        model_size_params = {"num_heads": 10, "num_layers": 4, "hidden_dim_factor": 20}
    elif model_size == "medium":
        model_size_params = {"num_heads": 12, "num_layers": 6, "hidden_dim_factor": 28}
    elif model_size == "large":
        model_size_params = {"num_heads": 16, "num_layers": 8, "hidden_dim_factor": 32}

    model = Weatherformer(
        input_dim=TOTAL_WEATHER_VARS, output_dim=TOTAL_WEATHER_VARS, **model_size_params
    ).to(DEVICE)

    logging.info(str(model))

    model, losses = training_loop(
        model,
        args.batch_size,
        num_input_features=args.n_input_features,
        num_output_features=TOTAL_WEATHER_VARS - args.n_input_features,
        num_epochs=args.n_epochs,
        init_lr=args.init_lr,
        num_warmup_epochs=args.n_warmup_epochs,
        decay_factor=args.decay_factor,
    )
