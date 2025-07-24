from typing import Dict, Tuple
import torch
import torch.nn as nn
import logging
import random

from src.base_trainer.base_trainer import BaseTrainer
from src.pretraining.models.mlp import MLP
from src.utils.constants import TOTAL_WEATHER_VARS, MAX_CONTEXT_LENGTH
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set seed for reproducibility
random.seed(1234)
torch.manual_seed(1234)


class MLPTrainer(BaseTrainer):
    """
    Simple MLP trainer.
    Input: all 31 weather features
    Output: predict 6 specific features [1, 2, 7, 8, 11, 29]
    Loss: MSE between predicted and actual target features + per-dim MSE
    """

    def __init__(self, model: MLP, max_len: int = MAX_CONTEXT_LENGTH, **kwargs):
        super().__init__(model, **kwargs)
        self.criterion = nn.MSELoss(reduction="none")  # For per-feature computation
        self.masked_features = [1, 2, 7, 8, 11, 29]
        self.max_len = max_len

        # Add per-feature MSE tracking
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                **{f"feature_{feat}_mse": [] for feat in self.masked_features},
            },
            "val": {
                "total_loss": [],
                **{f"feature_{feat}_mse": [] for feat in self.masked_features},
            },
        }

    def compute_train_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,  # Ignored
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss with per-feature MSE."""
        # Truncate sequences to max_len
        data = data[:, -self.max_len :]
        coords = coords  # Keep as is
        year = year[:, -self.max_len :]
        interval = interval  # Keep as is

        # Forward pass - gets all input, outputs 6 features
        output = self.model(data, coords, year, interval)

        # Target is the actual values of the masked features
        target = data[:, :, self.masked_features]

        # Per-feature MSE: [batch, seq, 6]
        feature_mse = self.criterion(output, target)

        # Mean over batch and sequence: [6]
        feature_losses = feature_mse.mean(dim=(0, 1))

        # Overall loss is mean of all features
        total_loss = feature_losses.mean()

        # Create result dict
        result = {"total_loss": total_loss}
        for i, feature_idx in enumerate(self.masked_features):
            result[f"feature_{feature_idx}_mse"] = feature_losses[i]

        return result

    def compute_validation_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,  # Ignored
    ) -> Dict[str, torch.Tensor]:
        """Compute validation loss with per-feature MSE."""
        with torch.no_grad():
            # Truncate sequences to max_len
            data = data[:, -self.max_len :]
            coords = coords  # Keep as is
            year = year[:, -self.max_len :]
            interval = interval  # Keep as is

            # Forward pass
            output = self.model(data, coords, year, interval)

            # Target
            target = data[:, :, self.masked_features]

            # Per-feature MSE: [batch, seq, 6]
            feature_mse = self.criterion(output, target)

            # Mean over batch and sequence: [6]
            feature_losses = feature_mse.mean(dim=(0, 1))

            # Overall loss
            total_loss = feature_losses.mean()

            # Create result dict
            result = {"total_loss": total_loss}
            for i, feature_idx in enumerate(self.masked_features):
                result[f"feature_{feature_idx}_mse"] = feature_losses[i]

        return result

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders."""
        train_loader = streaming_dataloader(
            self.batch_size,
            split="train",
            shuffle=shuffle,
            masking_function="weatherbert",  # Ignored anyway
            masking_prob=0.5,
            n_masked_features=len(self.masked_features),
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = streaming_dataloader(
            self.batch_size,
            split="validation",
            shuffle=False,
            masking_function="weatherbert",  # Ignored anyway
            masking_prob=0.5,
            n_masked_features=len(self.masked_features),
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_loader, val_loader


def mlp_training_loop(args_dict):
    """MLP training loop."""
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)
    max_len = args_dict.get("max_len", 52)  # Configurable max length

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize MLP model
    model = MLP(
        weather_dim=TOTAL_WEATHER_VARS,
        device=device,
        hidden_dim=512,
        max_len=max_len,  # Pass custom max_len
    ).to(device)

    if rank == 0:
        logging.info(f"MLP model with max_len={max_len}")
        logging.info(str(model))

    trainer = MLPTrainer(
        model=model,
        max_len=max_len,  # Pass max_len to trainer
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        resume_from_checkpoint=args_dict["resume_from_checkpoint"],
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train(use_optimal_lr=args_dict["use_optimal_lr"])
