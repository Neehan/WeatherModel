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

        # Training state
        self.logger = logging.getLogger(__name__)

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
            "losses": {"train": [], "val": []},
        }
        self.model_dir = DATA_DIR + "trained_models/pretraining/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Load pretrained model if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            pretrained_model = torch.load(pretrained_model_path, weights_only=False)
            self.model.load_pretrained(pretrained_model)

    # --- Abstract Methods ---
    @abstractmethod
    def create_feature_mask(
        self, batch_size: int, seq_len: int, n_features: int
    ) -> torch.Tensor:
        """
        Create a feature mask for the current training step.
        Children classes must implement this method based on their specific masking strategy.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            n_features: Number of weather features

        Returns:
            Boolean mask tensor
        """
        pass

    @abstractmethod
    def compute_train_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss for a batch.
        Children classes must implement this method.

        Args:
            weather: Weather data tensor
            coords: Coordinate tensor
            year: Year tensor
            interval: Interval tensor
            feature_mask: Feature mask tensor

        Returns:
            Loss tensor
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
    ) -> torch.Tensor:
        """
        Compute validation loss for a batch.
        Children classes must implement this method.

        Args:
            weather: Weather data tensor
            coords: Coordinate tensor
            year: Year tensor
            interval: Interval tensor
            feature_mask: Feature mask tensor

        Returns:
            Loss tensor
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
        total_loss = 0
        loader_len = 0

        self.logger.info(f"Started training epoch.")

        start_time = time.time()

        for weather, coords, year, interval, feature_mask in loader:
            weather = weather.to(self.device)
            coords = coords.to(self.device)
            year = year.to(self.device)
            interval = interval.to(self.device)
            feature_mask = feature_mask.to(self.device)

            self.optimizer.zero_grad()

            loss = self.compute_train_loss(
                weather, coords, year, interval, feature_mask
            )

            total_loss += loss.item()
            loader_len += 1

            if time.time() - start_time > self.log_interval_seconds and DRY_RUN:
                self.logger.info(f"Train loss: {loss.item():.3f}")
                start_time = time.time()

            # Backward pass
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        return total_loss / loader_len

    def validate_epoch(self, loader) -> float:
        """Validate the model for one epoch."""
        self.model.eval()
        total_loss = 0
        loader_len = 0

        self.logger.info(f"Started validation epoch.")

        for weather, coords, year, interval, feature_mask in loader:
            weather = weather.to(self.device)
            coords = coords.to(self.device)
            year = year.to(self.device)
            interval = interval.to(self.device)
            feature_mask = feature_mask.to(self.device)

            with torch.no_grad():
                loss = self.compute_validation_loss(
                    weather, coords, year, interval, feature_mask
                )

            total_loss += loss.item()
            loader_len += 1

        return total_loss / loader_len

    def save_model(self, epoch: int):
        """Save the model at the current epoch."""

        torch.save(
            self.model,
            self.model_dir + f"{self.model.name}_epoch_{epoch}.pth",
        )
        torch.save(
            self.model,
            self.model_dir + f"{self.model.name}_latest.pth",
        )

    def save_output_json(self):
        """Save the output JSON containing model config and losses."""
        filename = f"{self.model.name}_output.json"
        with open(self.model_dir + filename, "w") as f:
            json.dump(self.output_json, f, indent=2)

    def train(self, num_epochs: int) -> Tuple[BaseModel, Dict[str, List[float]]]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Tuple of (trained_model, losses_dict)
        """
        for epoch in range(num_epochs):
            train_loader = streaming_dataloader(
                self.batch_size,
                split="train",
                shuffle=True,
                masking_function=self.create_feature_mask,
            )

            test_loader = streaming_dataloader(
                self.batch_size,
                split="validation",
                shuffle=False,
                masking_function=self.create_feature_mask,
            )

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(test_loader)

            self.output_json["losses"]["train"].append(train_loss)
            self.output_json["losses"]["val"].append(val_loss)

            self.logger.info(
                f"Epoch [{epoch+1} / {num_epochs}]: Train loss: {train_loss:.3f} Validation loss: {val_loss:.3f}"
            )

            if epoch % 2 == 1 or epoch == num_epochs - 1:
                self.save_model(epoch)

            self.save_output_json()

        return self.model, self.output_json
