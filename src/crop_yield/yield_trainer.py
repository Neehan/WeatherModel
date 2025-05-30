import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from src.models.base_model import BaseModel
from src.pretraining.base.base_trainer import BaseTrainer
from src.pretraining.base.find_optimal_lr import find_optimal_lr
from src.utils.constants import DATA_DIR, DEVICE, TOTAL_WEATHER_VARS
from src.crop_yield.weatherbert_yield_model import WeatherBERTYieldModel
from src.crop_yield.weatherformer_yield_model import WeatherFormerYieldModel
from src.crop_yield.base.yield_dataloader import get_train_test_loaders
import os


class YieldTrainer(BaseTrainer):
    """
    Trainer class for crop yield prediction models.
    Provides training infrastructure for yield prediction tasks.
    """

    def __init__(
        self,
        model: BaseModel,
        batch_size: int,
        init_lr: float = 0.0009,
        num_warmup_epochs: int = 2,
        decay_factor: float = 0.95,
        log_interval_seconds: int = 10,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            init_lr=init_lr,
            num_warmup_epochs=num_warmup_epochs,
            decay_factor=decay_factor,
            log_interval_seconds=log_interval_seconds,
            pretrained_model_path=pretrained_model_path,
        )

        # Override criterion for yield prediction
        self.criterion = nn.MSELoss()
        self.model_dir = DATA_DIR + "trained_models/crop_yield/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Override the output_json structure for yield prediction
        self.output_json["losses"] = {
            "train": {"total_loss": []},
            "val": {"total_loss": []},
        }

    def get_model_name(self) -> str:
        """Get the model name for saving."""
        return f"{self.model.name}"

    def compute_train_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        This method is required by BaseTrainer but not used for yield prediction.
        Yield training uses custom train_epoch method instead.
        """
        raise NotImplementedError("Yield training uses custom train_epoch method")

    def compute_validation_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        This method is required by BaseTrainer but not used for yield prediction.
        Yield validation uses custom validate_epoch method instead.
        """
        raise NotImplementedError("Yield validation uses custom validate_epoch method")

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
        self,
        train_loader,
        test_loader,
        num_epochs: int = 20,
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
        # Find optimal learning rate if enabled
        # self.logger.info("Finding optimal learning rate...")
        # optimal_lr = find_optimal_lr(self, train_loader)

        # # Update optimizer with optimal learning rate
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = optimal_lr
        # self.logger.info(f"Updated learning rate to optimal value: {optimal_lr:.6f}")

        best_test_rmse = float("inf")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_rmse = self.validate_epoch(test_loader)

            self.output_json["losses"]["train"]["total_loss"].append(train_loss)
            self.output_json["losses"]["val"]["total_loss"].append(test_rmse)

            best_test_rmse = min(test_rmse, best_test_rmse)

            self.logger.info(
                f"[{epoch+1} / {num_epochs}] Train RMSE: {train_loss:.3f}, Test RMSE best: {best_test_rmse:.3f}, current: {test_rmse:.3f}"
            )

            if epoch % 2 == 1 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch, test_rmse)

            self.save_output_json()

        return self.model, best_test_rmse

    @staticmethod
    def create_yield_model(
        model_type: str,
        model_size_params: dict,
        n_past_years: int,
        pretrained_model_path: Optional[str] = None,
    ):
        """Create and return the appropriate yield prediction model."""

        if model_type.lower() == "weatherformer":
            yield_model_cls = WeatherFormerYieldModel
        elif model_type.lower() == "weatherbert":
            yield_model_cls = WeatherBERTYieldModel
        else:
            raise ValueError(f"Unknown yield model type: {model_type}")

        mlp_input_dim = (
            TOTAL_WEATHER_VARS * 52 * (n_past_years + 1)
        )  # 52 weeks * n_past_years + 1

        yield_model = yield_model_cls(
            name=f"{model_type}_yield",
            mlp_input_dim=mlp_input_dim,
            weather_dim=TOTAL_WEATHER_VARS,
            output_dim=TOTAL_WEATHER_VARS,
            device=DEVICE,
            **model_size_params,
        )

        # Load pretrained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            pretrained_model = torch.load(pretrained_model_path, weights_only=False)
            yield_model.load_pretrained(pretrained_model)

        return yield_model

    @classmethod
    def cross_validation_training_loop(cls, soybean_df, args_dict: dict) -> float:
        """
        Main yield prediction cross-validation training loop.

        Args:
            soybean_df: DataFrame containing soybean yield data
            args_dict: Dictionary containing training arguments

        Returns:
            Average best validation RMSE across all folds
        """
        logger = logging.getLogger(__name__)

        soybean_states = set(soybean_df["State"].values)
        logger.info(f"Loaded soybean dataset with {len(soybean_df)} rows")
        logger.info(f"Available states: {sorted(list(soybean_states))}")

        total_best_val_loss = 0.0
        n_folds = args_dict["n_cross_validation_folds"]

        for fold in range(n_folds):
            logger.info(f"Starting cross-validation fold {fold + 1}/{n_folds}")

            # Randomly select test states for this fold
            test_states = np.random.choice(
                np.array(sorted(list(soybean_states))), size=2, replace=False
            )
            logger.info(f"Testing on states: {test_states}")

            # Create data loaders
            train_loader, test_loader = get_train_test_loaders(
                soybean_df,
                test_states,
                n_past_years=args_dict["n_past_years"],
                batch_size=args_dict["batch_size"],
            )

            # Create the model
            pretrained_path = args_dict.get("pretrained_model_path")
            model = cls.create_yield_model(
                model_type=args_dict["model"],
                model_size_params=args_dict["model_size_params"],
                n_past_years=args_dict["n_past_years"],
                pretrained_model_path=pretrained_path,
            )

            model = model.to(DEVICE)
            logger.info(f"Created model: {model.name}")
            logger.info(f"Total parameters: {model.total_params_formatted()}")
            logger.info(f"Model architecture:\n{model}")

            # Create trainer
            trainer = cls(
                model=model,
                batch_size=args_dict["batch_size"],
                init_lr=args_dict["init_lr"],
                num_warmup_epochs=args_dict["n_warmup_epochs"],
                decay_factor=args_dict["lr_decay_factor"],
            )

            # Train the model
            trained_model, best_val_rmse = trainer.train(
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=args_dict["n_epochs"],
            )

            total_best_val_loss += best_val_rmse
            logger.info(f"Fold {fold + 1} best validation RMSE: {best_val_rmse:.3f}")

        # Calculate and log average performance
        avg_best_val_loss = total_best_val_loss / n_folds
        # Convert RMSE to Bu/Acre (using conversion factor from original code)
        avg_best_val_loss_bu_acre = avg_best_val_loss * 11.03

        logger.info(f"Average best validation RMSE: {avg_best_val_loss:.3f}")
        logger.info(
            f"Average best validation loss: {avg_best_val_loss_bu_acre:.3f} Bu/Acre"
        )

        return avg_best_val_loss
