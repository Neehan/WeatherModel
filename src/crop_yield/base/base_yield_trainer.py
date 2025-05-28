import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Tuple
from src.pretraining.base.base_trainer import BaseTrainer


class BaseYieldTrainer(BaseTrainer):
    """
    Base trainer class for crop yield prediction models.
    Provides training infrastructure for yield prediction tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        init_lr: float = 0.0009,
        num_warmup_epochs: int = 2,
        decay_factor: float = 0.95,
        log_interval_seconds: int = 10,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            init_lr=init_lr,
            num_warmup_epochs=num_warmup_epochs,
            decay_factor=decay_factor,
            log_interval_seconds=log_interval_seconds,
        )

        # Override criterion for yield prediction
        self.criterion = nn.MSELoss()

    def get_model_name(self) -> str:
        """Get the model name for saving."""
        return f"yield_model_{self.total_params_formatted}"

    def train_epoch(self, train_loader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        loader_len = len(train_loader)

        self.logger.info("Started training epoch.")
        start_time = time.time()

        for i, (input_data, y) in enumerate(train_loader):
            # Zero the gradients
            self.optimizer.zero_grad()
            input_data = [x.to(self.device) for x in input_data]
            y = y.to(self.device)

            # Forward pass
            outputs = self.model(input_data)
            loss = self.criterion(outputs, y)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            if time.time() - start_time > self.log_interval_seconds:
                self.logger.info(f"Train loss: {loss.item():.3f}")
                start_time = time.time()

        self.scheduler.step()

        # Convert to RMSE
        running_loss /= loader_len
        running_loss = math.sqrt(running_loss)
        return running_loss

    def validate_epoch(self, test_loader) -> float:
        """Validate the model for one epoch."""
        self.model.eval()
        mse_total = 0.0

        self.logger.info("Started validation epoch.")

        with torch.no_grad():
            for input_data, y in test_loader:
                input_data = [x.to(self.device) for x in input_data]
                y = y.to(self.device)

                # Forward pass
                outputs = self.model(input_data)

                # Compute the mean squared error
                mse = F.mse_loss(outputs, y)

                # Accumulate the MSE over all batches
                mse_total += mse.item()

        # Compute the RMSE
        rmse = math.sqrt(mse_total / len(test_loader))
        return rmse

    def train(
        self, train_loader, test_loader, num_epochs: int = 20
    ) -> Tuple[nn.Module, float]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs to train

        Returns:
            Tuple of (trained_model, best_test_rmse)
        """
        best_test_rmse = float("inf")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_rmse = self.validate_epoch(test_loader)

            self.output_json["losses"]["train"].append(train_loss)
            self.output_json["losses"]["val"].append(test_rmse)

            best_test_rmse = min(test_rmse, best_test_rmse)

            self.logger.info(
                f"[{epoch+1} / {num_epochs}] Train RMSE: {train_loss:.3f}, Test RMSE best: {best_test_rmse:.3f}, current: {test_rmse:.3f}"
            )

            if epoch % 2 == 1 or epoch == num_epochs - 1:
                self.save_model(epoch)

            self.save_output_json()

        return self.model, best_test_rmse
