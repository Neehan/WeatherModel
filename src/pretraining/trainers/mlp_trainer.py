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
    MLP trainer that implements custom masking for specific weather features.
    Always masks features [7, 8, 11, 1, 2, 29] and computes per-feature MSE.
    """

    def __init__(
        self,
        model: MLP,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.criterion = nn.MSELoss(
            reduction="none"
        )  # Use reduction='none' for per-feature loss

        # Fixed mask features - always mask these features
        self.masked_features = [7, 8, 11, 1, 2, 29]

        # Override the losses collected to include per-feature MSE
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

        self.output_json["model_config"]["masked_features"] = self.masked_features

        # Create the constant mask once in constructor
        # Shape: (1, seq_len, n_features) - batch_size=1 since we'll expand
        self.mask_template = torch.zeros(
            1, MAX_CONTEXT_LENGTH, TOTAL_WEATHER_VARS, dtype=torch.bool
        )

        # Set mask to True for the specified features using vectorized indexing
        self.mask_template[:, :, self.masked_features] = True

    def compute_train_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,  # This will be ignored
    ) -> Dict[str, torch.Tensor]:
        """Compute MLP training loss using custom mask and per-feature MSE."""
        batch_size, seq_len, n_features = data.shape
        device = data.device

        # Expand the mask template to current batch size and move to device
        custom_mask = (
            self.mask_template[:, :seq_len, :n_features]
            .expand(batch_size, -1, -1)
            .to(device)
        )

        # Forward pass with custom mask
        output = self.model(
            data, coords, year, interval, weather_feature_mask=custom_mask
        )

        # Vectorized per-feature MSE computation
        per_feature_losses = {}

        # Extract all masked features at once: [batch_size, seq_len, num_masked_features]
        target_features = data[:, :, self.masked_features]
        predicted_features = output[:, :, self.masked_features]

        # Compute MSE per feature: [batch_size, seq_len, num_masked_features]
        feature_mse = self.criterion(predicted_features, target_features)

        # Mean over batch and sequence dimensions: [num_masked_features]
        feature_losses = feature_mse.mean(dim=(0, 1))

        # Create per-feature loss dict
        for i, feature_idx in enumerate(self.masked_features):
            per_feature_losses[f"feature_{feature_idx}_mse"] = feature_losses[i]

        # Overall loss is mean of all feature losses
        total_loss = feature_losses.mean()

        result = {"total_loss": total_loss}
        result.update(per_feature_losses)
        return result

    def compute_validation_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,  # This will be ignored
    ) -> Dict[str, torch.Tensor]:
        """Compute MLP validation loss using custom mask and per-feature MSE."""
        batch_size, seq_len, n_features = data.shape
        device = data.device

        # Expand the mask template to current batch size and move to device
        custom_mask = (
            self.mask_template[:, :seq_len, :n_features]
            .expand(batch_size, -1, -1)
            .to(device)
        )

        with torch.no_grad():
            # Forward pass with custom mask
            output = self.model(
                data, coords, year, interval, weather_feature_mask=custom_mask
            )

            # Vectorized per-feature MSE computation
            per_feature_losses = {}

            # Extract all masked features at once: [batch_size, seq_len, num_masked_features]
            target_features = data[:, :, self.masked_features]
            predicted_features = output[:, :, self.masked_features]

            # Compute MSE per feature: [batch_size, seq_len, num_masked_features]
            feature_mse = self.criterion(predicted_features, target_features)

            # Mean over batch and sequence dimensions: [num_masked_features]
            feature_losses = feature_mse.mean(dim=(0, 1))

            # Create per-feature loss dict
            for i, feature_idx in enumerate(self.masked_features):
                per_feature_losses[f"feature_{feature_idx}_mse"] = feature_losses[i]

            # Overall loss is mean of all feature losses
            total_loss = feature_losses.mean()

        result = {"total_loss": total_loss}
        result.update(per_feature_losses)
        return result

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation."""

        # Use weatherbert masking function but we'll override the mask anyway
        train_loader = streaming_dataloader(
            self.batch_size,
            split="train",
            shuffle=shuffle,
            masking_function="weatherbert",  # We'll ignore this mask
            masking_prob=0.5,  # Doesn't matter since we override
            n_masked_features=len(self.masked_features),
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = streaming_dataloader(
            self.batch_size,
            split="validation",
            shuffle=False,
            masking_function="weatherbert",  # We'll ignore this mask
            masking_prob=0.5,  # Doesn't matter since we override
            n_masked_features=len(self.masked_features),
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_loader, val_loader


def mlp_training_loop(args_dict):
    """
    MLP training loop using the MLPTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize MLP model
    model = MLP(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        hidden_dim=128,  # Fixed 128 as requested
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = MLPTrainer(
        model=model,
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
