from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from src.model import utils
from src.model.constants import *


def find_optimal_lr(
    model,
    batch_size,
    num_input_features,
    num_output_features,
    init_lr,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params/10**6:.2f}M")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    data_loader_dir = DATA_DIR + "nasa_power/pytorch/"

    train_indices = DRY_RUN_TRAIN_PART_IDS
    train_loader_paths = [
        data_loader_dir + f"weather_dataset_{frequency}_{i}.pt"
        for i in train_indices
        for frequency in ["monthly", "weekly", "daily"]
    ]

    # Find the optimal learning rate
    train_loader = utils.streaming_dataloader(
        train_loader_paths,
        batch_size,
        shuffle=True,
        split="train",
        lr_finder=True,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
    )
    optimal_lr = utils.find_optimal_lr(
        model, criterion, optimizer, train_loader, DEVICE
    )
    logging.info(f"optimal learning rate: {optimal_lr:.6f}")

    return optimal_lr
