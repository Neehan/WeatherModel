import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
import os

from src.base_trainer.base_trainer import BaseTrainer
from src.yield_pretraining.models.seq_model import SeqModel
from src.yield_pretraining.dataloader.seq_dataloader import SeqDataloader
from src.utils.constants import DATA_DIR


class SeqTrainer(BaseTrainer):
    """Trainer for the sequence-based yield prediction model."""

    def __init__(self, model: SeqModel, dataloader: SeqDataloader, **kwargs):

        # Call parent constructor with all kwargs
        super().__init__(model=model, **kwargs)

        # Override model directory for yield prediction
        if self.rank == 0:
            self.model_dir = DATA_DIR + "trained_models/yield_pretraining/"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        # Store dataloader
        self.dataloader = dataloader

        # Initialize loss function
        self.criterion = nn.MSELoss(
            reduction="none"
        )  # Use reduction='none' for masking

        # override the losses collected
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
            },
            "val": {
                "total_loss": [],
            },
        }

    def get_dataloaders(
        self, shuffle: bool = True, cross_validation_k: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders."""
        return self.dataloader.get_dataloaders(shuffle=shuffle)

    def compute_train_loss(
        self, years, coords, periods, yields, yield_mask, padding_mask
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for a batch."""
        predictions = self.model(
            years, coords, periods, yields, yield_mask, padding_mask
        )

        # Compute loss only on masked (unknown) yields
        loss_per_element = self.criterion(predictions, yields)

        # Apply inverse mask to compute loss only on unknown yields (yield_mask=False)
        masked_loss = loss_per_element * (~yield_mask).float()

        # Average over valid (non-masked) positions
        num_masked = (~yield_mask).float().sum()
        if num_masked > 0:
            loss = masked_loss.sum() / num_masked
        else:
            loss = torch.tensor(0.0, device=predictions.device)

        return {"total_loss": loss}

    def compute_validation_loss(
        self, years, coords, periods, yields, yield_mask, padding_mask
    ) -> Dict[str, torch.Tensor]:
        """Compute validation loss for a batch."""
        with torch.no_grad():
            predictions = self.model(
                years, coords, periods, yields, yield_mask, padding_mask
            )

            # Compute loss only on masked (unknown) yields
            loss_per_element = self.criterion(predictions, yields)

            # Apply inverse mask to compute loss only on unknown yields (yield_mask=False)
            masked_loss = loss_per_element * (~yield_mask).float()

            # Average over valid (non-masked) positions
            num_masked = (~yield_mask).float().sum()
            if num_masked > 0:
                loss = masked_loss.sum() / num_masked
            else:
                loss = torch.tensor(0.0, device=predictions.device)

            # return rmse in val
            return {"total_loss": loss**0.5}
