import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)
import torch.nn as nn


import numpy as np
import argparse

from src.model.train import training_loop
from src.model.weatherformer_v2 import Weatherformer
from src.model.bert_model import WeatherBERT
from src.model.bert_train import bert_training_loop
from src.model.find_optimal_lr import find_optimal_lr
from src.model.constants import *

np.random.seed(1234)
torch.manual_seed(1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument(
        "--n_input_features", help="number of input features", default=21, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of training epochs", default=75, type=int
    )
    parser.add_argument(
        "--n_feature_swaps",
        help="number of features to swap per batch",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_lr", help="initial learning rate", default=0.0005, type=float
    )
    parser.add_argument(
        "--n_warmup_epochs", help="number of warm-up epochs", default=10, type=float
    )
    parser.add_argument(
        "--decay_factor",
        help="exponential learning rate decay factor after warmup",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--model_size",
        help="model size small (2M), medium (8M), and large (56M)",
        default="small",
        type=str,
    )

    parser.add_argument(
        "--model_type",
        help="model type is weatherformer or bert",
        default="weatherformer",
        type=str,
    )

    parser.add_argument(
        "--mask_pcnt",
        help="percent to mask",
        default=0.15,
        type=float,
    )

    parser.add_argument(
        "--clip_gradients",
        help="enable gradient clipping",
        action="store_true",
    )

    parser.add_argument(
        "--find_optimal_lr",
        help="find optimal learning rate",
        action="store_true",
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
        model_size_params = {"num_heads": 16, "num_layers": 8, "hidden_dim_factor": 48}

    model_type = args.model_type.lower()

    if model_type == "weatherformer":
        model = Weatherformer(
            input_dim=TOTAL_WEATHER_VARS,
            output_dim=TOTAL_WEATHER_VARS,
            device=DEVICE,
            **model_size_params,
        ).to(DEVICE)
    elif model_type == "bert":
        model = WeatherBERT(
            input_dim=TOTAL_WEATHER_VARS,
            output_dim=TOTAL_WEATHER_VARS,
            device=DEVICE,
            **model_size_params,
        ).to(DEVICE)

    logging.info(str(model))

    if torch.cuda.device_count() > 1:
        logging.info(
            f"Found {torch.cuda.device_count()} GPUs. Using DataParallel class to parellelize training."
        )
        args.batch_size = args.batch_size * torch.cuda.device_count()
        model = nn.DataParallel(model)

    if args.find_optimal_lr:
        # Call the find_optimal_lr function before training
        optimal_lr = find_optimal_lr(
            model, args.batch_size, num_input_features=args.n_input_features
        )
        # Use the optimal learning rate for training
        args.init_lr = optimal_lr

    if model_type == "weatherformer":
        model, losses = training_loop(
            model,
            args.batch_size,
            num_input_features=args.n_input_features,
            num_output_features=TOTAL_WEATHER_VARS - args.n_input_features,
            num_epochs=args.n_epochs,
            init_lr=args.init_lr,
            num_warmup_epochs=args.n_warmup_epochs,
            decay_factor=args.decay_factor,
            num_feature_swaps=args.n_feature_swaps,
            enable_gradient_clipping=args.clip_gradients,
        )
    elif model_type == "bert":
        model, losses = bert_training_loop(
            model,
            args.batch_size,
            args.n_epochs,
            args.init_lr,
            args.n_warmup_epochs,
            args.decay_factor,
            args.mask_pcnt,
            enable_gradient_clipping=args.clip_gradients,
        )
