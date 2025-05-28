from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch_lr_finder import LRFinder
import logging

from src.pretraining.base.pretraining_dataloader import streaming_dataloader
from src.utils.constants import DEVICE

logger = logging.getLogger(__name__)


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
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-2)

    # Find the optimal learning rate
    train_loader = streaming_dataloader(
        batch_size,
        shuffle=True,
        split="train",
        num_input_features=num_input_features,
        num_output_features=num_output_features,
    )
    lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

    # Extract optimal LR from the history
    history = lr_finder.history
    lrs = history["lr"]
    losses = history["loss"]
    optimal_lr = lrs[losses.index(min(losses))]

    # Reset the model and optimizer to their initial state
    lr_finder.reset()
    logger.info(f"optimal learning rate: {optimal_lr:.6f}")

    return optimal_lr
