from typing import Union, TYPE_CHECKING
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import logging
import numpy as np

from src.utils.constants import DEVICE

if TYPE_CHECKING:
    from src.pretraining.base.base_trainer import BaseTrainer
    from src.crop_yield.base.base_yield_trainer import BaseYieldTrainer

logger = logging.getLogger(__name__)


def simple_lr_finder(
    trainer,
    dataloader,
    start_lr: float = 1e-6,
    end_lr: float = 1,
    num_iter: int = 100,
    is_pretraining: bool = True,
):
    """
    Simple learning rate finder that works with multi-input models.
    """
    # Store original lr
    original_lr = trainer.optimizer.param_groups[0]["lr"]

    # Create learning rate schedule
    lr_mult = (end_lr / start_lr) ** (1.0 / (num_iter - 1))

    lrs = []
    losses = []
    best_loss = None

    # Set starting learning rate
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = start_lr

    current_lr = start_lr

    # Create iterator
    data_iter = iter(dataloader)

    trainer.model.train()

    for i in tqdm(range(num_iter)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Zero gradients
        trainer.optimizer.zero_grad()

        if is_pretraining:
            # Pretraining format: (weather, coords, year, interval, feature_mask)
            weather, coords, year, interval, feature_mask = batch
            weather = weather.to(DEVICE)
            coords = coords.to(DEVICE)
            year = year.to(DEVICE)
            interval = interval.to(DEVICE)
            feature_mask = feature_mask.to(DEVICE)

            # Compute loss using trainer's method
            loss = trainer.compute_train_loss(
                weather, coords, year, interval, feature_mask
            )
        else:
            # Yield training format: (input_data, y)
            input_data, y = batch
            input_data = [x.to(DEVICE) for x in input_data]
            y = y.to(DEVICE)

            # Forward pass and compute loss
            outputs = trainer.model(input_data)
            loss = trainer.criterion(outputs, y)

        # Backward pass
        loss.backward()
        trainer.optimizer.step()

        # Record
        lrs.append(current_lr)
        loss_val = loss.item()
        losses.append(loss_val)

        # Track best loss
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val

        # Check for divergence
        if loss_val > 5 * best_loss:
            logger.info("Stopping early due to loss divergence")
            break

        # Update learning rate
        current_lr *= lr_mult
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = current_lr

    # Find optimal learning rate (steepest descent)
    gradients = np.gradient(losses)
    min_grad_idx = gradients.argmin()
    optimal_lr = lrs[min_grad_idx]

    # Restore original learning rate
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = original_lr

    return optimal_lr


def find_optimal_lr(
    trainer: "Union[BaseTrainer, BaseYieldTrainer]",
    dataloader: DataLoader,
    start_lr: float = 1e-6,
    end_lr: float = 1,
    num_iter: int = 100,
    is_pretraining: bool = True,
):
    """
    Find optimal learning rate for any trainer using learning rate range test.

    Args:
        trainer: Any trainer instance (BaseTrainer or BaseYieldTrainer subclass)
        dataloader: Training data loader
        start_lr: Minimum learning rate to test
        end_lr: Maximum learning rate to test
        num_iter: Number of iterations for the test

    Returns:
        Optimal learning rate
    """
    optimal_lr = simple_lr_finder(
        trainer, dataloader, start_lr, end_lr, num_iter, is_pretraining
    )
    logger.info(f"optimal learning rate: {optimal_lr:.6f}")
    return optimal_lr
