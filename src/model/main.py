import numpy as np
import logging
import torch
import argparse

from train import training_loop
from model import Weatherformer
from constants import *
import utils

np.random.seed(1234)
torch.manual_seed(1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument(
        "--n_input_features", help="number of input features", default=26, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of training epochs", default=10, type=int
    )
    parser.add_argument(
        "--init_lr", help="initial learning rate", default=0.0005, type=float
    )
    parser.add_argument(
        "--n_warmup_epochs", help="number of warm-up epochs", default=2, type=float
    )
    parser.add_argument(
        "--decay_factor",
        help="exponential learning rate decay factor after warmup",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--train-attn",
        dest="train_attn",
        action="store_true",
        help="train the attention layer",
    )
    parser.set_defaults(train_attn=False)

    args = parser.parse_args()

    model = Weatherformer(
        input_dim=TOTAL_WEATHER_VARS, output_dim=TOTAL_WEATHER_VARS
    ).to(DEVICE)
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
