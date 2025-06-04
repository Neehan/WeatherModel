from typing import Union, TYPE_CHECKING
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import logging
import numpy as np
from src.utils.constants import DRY_RUN

if TYPE_CHECKING:
    from src.base_trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def find_optimal_lr(
    trainer: "BaseTrainer",
    dataloader: DataLoader,
    start_lr: float = 1e-6,
    end_lr: float = 0.1,  # Conservative for transformers - can be increased if needed
    num_iter: int = 100 if not DRY_RUN else 5,
):
    """
    Find optimal learning rate using Leslie Smith's LR range test methodology.

    Based on "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017)
    and "A disciplined approach to neural network hyper-parameters" (Smith, 2018).

    Args:
        trainer: Any trainer instance (BaseTrainer or BaseYieldTrainer subclass)
        dataloader: Training data loader
        start_lr: Minimum learning rate to test (should be very small)
        end_lr: Maximum learning rate to test (conservative default for transformers)
        num_iter: Number of iterations for the test
        is_pretraining: Whether this is pretraining or fine-tuning

    Returns:
        Optimal learning rate (typically 1/10th of the steepest decline point)
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

    # Check if we're in distributed mode
    is_distributed = dist.is_initialized() if dist.is_available() else False

    for i in tqdm(range(num_iter)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Zero gradients
        trainer.optimizer.zero_grad()

        batch = [x.to(trainer.device) for x in batch]
        # Compute loss using trainer's method
        loss_dict = trainer.compute_train_loss(*batch)
        loss = loss_dict["total_loss"]
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

    # Wait for all GPUs to finish their LR search
    if is_distributed:
        dist.barrier()

    # Find optimal learning rate using Leslie Smith's methodology
    # Leslie Smith approach: find where loss decreases fastest, then back off

    # First, find where loss starts diverging (loss increases significantly)
    min_loss = min(losses)
    min_loss_idx = losses.index(min_loss)

    # Look for divergence point - where loss > 4 * min_loss
    diverge_idx = len(losses)  # Default to end if no divergence found
    for i in range(min_loss_idx, len(losses)):
        if losses[i] > 4 * min_loss:
            diverge_idx = i
            break

    # Calculate gradients (rate of change of loss w.r.t. iteration)
    gradients = np.gradient(losses)

    # Find the most negative gradient (steepest decline) before divergence
    # Only consider the region up to divergence point
    search_region = gradients[:diverge_idx]
    if len(search_region) > 0:
        steepest_idx = np.argmin(search_region)
        steepest_lr = lrs[steepest_idx]

        # Leslie Smith recommendation: use LR that's ~1/10th of where steep decline occurs
        # This ensures we're in the steep part but not too close to divergence
        optimal_lr = steepest_lr / 10

        # Ensure it's not too small
        if optimal_lr < start_lr * 10:
            optimal_lr = start_lr * 10

    else:
        optimal_lr = start_lr * 10
        logger.warning("No clear steepest decline found, using conservative default")

    # Restore original learning rate
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = original_lr

    return optimal_lr
