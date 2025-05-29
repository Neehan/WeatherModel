import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import logging
import os
import json
from typing import Dict, List, Any, Tuple, Optional
import time
from src.pretraining.base.pretraining_dataloader import streaming_dataloader
from src.pretraining.base.find_optimal_lr import find_optimal_lr
from src.utils import utils
from src.utils.constants import DATA_DIR, DEVICE, DRY_RUN
from src.models.base_model import BaseModel


class BaseTrainer(ABC):
    """
    Base trainer class that provides common training infrastructure.
    Children classes need to implement feature masking and loss computation methods.
    """

    def __init__(
        self,
        model: BaseModel,
        batch_size: int,
        init_lr: float = 1e-4,
        num_warmup_epochs: int = 5,
        decay_factor: float = 0.95,
        log_interval_seconds: int = 10,
        pretrained_model_path: Optional[str] = None,
        masking_function: Optional[str] = None,
        masking_prob: float = 0.15,
        n_masked_features: int = 1,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.num_warmup_epochs = num_warmup_epochs
        self.decay_factor = decay_factor
        self.device = DEVICE
        self.log_interval_seconds = log_interval_seconds
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        self.scheduler = utils.get_scheduler(
            self.optimizer, num_warmup_epochs, decay_factor
        )
        self.masking_function = masking_function
        # Training state
        self.logger = logging.getLogger(__name__)

        self.masking_prob = masking_prob
        self.n_masked_features = n_masked_features

        self.logger.info(
            f"Total number of parameters: {self.model.total_params_formatted()}"
        )

        self.output_json = {
            "model_config": {
                "total_params": self.model.total_params(),
                "batch_size": batch_size,
                "init_lr": init_lr,
                "num_warmup_epochs": num_warmup_epochs,
                "decay_factor": decay_factor,
                "model_layers": str(self.model),
            },
            "losses": {"train": {"total_loss": []}, "val": {"total_loss": []}},
        }
        self.model_dir = DATA_DIR + "trained_models/pretraining/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Load pretrained model if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            pretrained_model = torch.load(pretrained_model_path, weights_only=False)
            self.model.load_pretrained(pretrained_model)

        self.start_epoch = 0
        # Resume from checkpoint if provided
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            self.load_checkpoint(resume_from_checkpoint)

    @abstractmethod
    def compute_train_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for a batch and update the output JSON.
        Children classes must implement this method.

        Args:
            weather: Weather data tensor
            coords: Coordinate tensor
            year: Year tensor
            interval: Interval tensor
            feature_mask: Feature mask tensor

        Returns:
            Loss tensor (called total_loss) and other loss tensors
        """
        pass

    @abstractmethod
    def compute_validation_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validation loss for a batch and update the output JSON.
        Children classes must implement this method.

        Args:
            weather: Weather data tensor
            coords: Coordinate tensor
            year: Year tensor
            interval: Interval tensor
            feature_mask: Feature mask tensor

        Returns:
            Loss tensor (called total_loss) and other loss tensors
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name for saving. Override in children classes."""
        pass

    # --- Internal Implementation ---

    def train_epoch(self, loader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        loader_len = 0
        total_loss_dict = {key: 0.0 for key in self.output_json["losses"]["train"]}

        self.logger.info(f"Started training epoch.")

        start_time = time.time()

        for weather, coords, year, interval, feature_mask in loader:
            weather = weather.to(self.device)
            coords = coords.to(self.device)
            year = year.to(self.device)
            interval = interval.to(self.device)
            feature_mask = feature_mask.to(self.device)

            self.optimizer.zero_grad()

            loss_dict = self.compute_train_loss(
                weather, coords, year, interval, feature_mask
            )
            loss = loss_dict["total_loss"]

            # Accumulate losses and its components
            for key in loss_dict:
                total_loss_dict[key] += loss_dict[key].item()

            loader_len += 1

            if time.time() - start_time > self.log_interval_seconds and DRY_RUN:
                self.logger.info(f"Train loss: {loss.item():.3f}")
                start_time = time.time()

            # Backward pass
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        for key in self.output_json["losses"]["train"]:
            self.output_json["losses"]["train"][key] = total_loss_dict[key] / loader_len

        return total_loss_dict["total_loss"]

    def validate_epoch(self, loader) -> float:
        """Validate the model for one epoch."""
        self.model.eval()
        loader_len = 0
        total_loss_dict = {key: 0.0 for key in self.output_json["losses"]["val"]}

        self.logger.info(f"Started validation epoch.")

        for weather, coords, year, interval, feature_mask in loader:
            weather = weather.to(self.device)
            coords = coords.to(self.device)
            year = year.to(self.device)
            interval = interval.to(self.device)
            feature_mask = feature_mask.to(self.device)

            with torch.no_grad():
                loss_dict = self.compute_validation_loss(
                    weather, coords, year, interval, feature_mask
                )

            loader_len += 1

            # Accumulate losses and its components
            for key in loss_dict:
                total_loss_dict[key] += loss_dict[key].item()

            for key in self.output_json["losses"]["val"]:
                self.output_json["losses"]["val"][key] = (
                    total_loss_dict[key] / loader_len
                )

        return total_loss_dict["total_loss"]

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save a complete checkpoint including model, optimizer, scheduler, and training state."""
        checkpoint = {
            "epoch": epoch + 1,  # Next epoch to resume from
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "output_json": self.output_json,
        }

        # Save epoch-specific checkpoint
        checkpoint_path = (
            self.model_dir + f"{self.model.name}_epoch_{epoch}_checkpoint.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_checkpoint_path = (
            self.model_dir + f"{self.model.name}_latest_checkpoint.pth"
        )
        torch.save(checkpoint, latest_checkpoint_path)

        # Also save the model separately for compatibility
        torch.save(self.model, self.model_dir + f"{self.model.name}_epoch_{epoch}.pth")
        torch.save(self.model, self.model_dir + f"{self.model.name}_latest.pth")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        if "output_json" in checkpoint:
            self.output_json = checkpoint["output_json"]
        self.logger.info(
            f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {self.start_epoch}"
        )

    def save_output_json(self):
        """Save the output JSON containing model config and losses."""
        filename = f"{self.model.name}_output.json"
        with open(self.model_dir + filename, "w") as f:
            json.dump(self.output_json, f, indent=2)

    def train(
        self, num_epochs: int, use_optimal_lr: bool = True
    ) -> Tuple[BaseModel, Dict[str, List[float]]]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Tuple of (trained_model, losses_dict)
        """
        # Find optimal learning rate if enabled and not resuming from checkpoint
        if use_optimal_lr and self.start_epoch == 0:
            self.logger.info("Finding optimal learning rate...")
            train_loader = streaming_dataloader(
                self.batch_size,
                split="train",
                shuffle=True,
                masking_function=self.masking_function,
                masking_prob=self.masking_prob,
                n_masked_features=self.n_masked_features,
            )
            optimal_lr = find_optimal_lr(self, train_loader)

            # Update optimizer with optimal learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = optimal_lr
            self.logger.info(
                f"Updated learning rate to optimal value: {optimal_lr:.6f}"
            )

        for epoch in range(self.start_epoch, num_epochs):
            train_loader = streaming_dataloader(
                self.batch_size,
                split="train",
                shuffle=True,
                masking_function=self.masking_function,
                masking_prob=self.masking_prob,
                n_masked_features=self.n_masked_features,
            )

            test_loader = streaming_dataloader(
                self.batch_size,
                split="validation",
                shuffle=False,
                masking_function=self.masking_function,
                masking_prob=self.masking_prob,
                n_masked_features=self.n_masked_features,
            )

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(test_loader)

            self.logger.info(
                f"Epoch [{epoch+1} / {num_epochs}]: Train loss: {train_loss:.3f} Validation loss: {val_loss:.3f}"
            )

            if epoch % 2 == 1 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch, val_loss)

            self.save_output_json()

        return self.model, self.output_json
