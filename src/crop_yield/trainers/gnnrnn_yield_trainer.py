import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from src.crop_yield.models.gnnrnn_yield_model import GNNRNNYieldModel
from src.crop_yield.dataloader.gnnrnn_dataloader import get_gnnrnn_dataloaders
from src.crop_yield.dataloader.yield_dataloader import read_soybean_dataset
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    TEST_YEARS,
    EXTREME_YEARS,
)


class GNNRNNYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for GNN-RNN crop yield prediction models.
    Inherits from WeatherBERTYieldTrainer but overrides to handle GNN-specific training.
    """

    def __init__(
        self,
        crop_df: pd.DataFrame,
        n_past_years: int,
        n_train_years: int,
        beta: float,
        use_cropnet: bool,
        crop_type: str,
        test_year: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            crop_df=crop_df,
            n_past_years=n_past_years,
            n_train_years=n_train_years,
            beta=beta,
            use_cropnet=use_cropnet,
            crop_type=crop_type,
            test_year=test_year,
            **kwargs,
        )
        self.logger = logging.getLogger(__name__)

    def train_one_epoch(self, train_loader, epoch_idx):
        """Train one epoch with GNN-RNN specific handling"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # For GNN-RNN, we need to handle the data differently
        if hasattr(train_loader, "sequences"):
            # Direct dataset access
            dataset = train_loader
            batch_size = min(32, len(dataset))

            for i in range(0, len(dataset), batch_size):
                batch_data = []
                for j in range(i, min(i + batch_size, len(dataset))):
                    batch_data.append(dataset[j])

                if not batch_data:
                    continue

                # Collate batch
                weather_batch = torch.stack(
                    [torch.FloatTensor(item["weather"]) for item in batch_data]
                )
                soil_batch = torch.stack(
                    [torch.FloatTensor(item["soil"]) for item in batch_data]
                )
                coords_batch = torch.stack(
                    [torch.FloatTensor(item["coords"]) for item in batch_data]
                )
                past_yields_batch = torch.stack(
                    [torch.FloatTensor(item["past_yields"]) for item in batch_data]
                )
                targets_batch = torch.FloatTensor(
                    [item["target_yield"] for item in batch_data]
                )

                # Move to device
                weather_batch = weather_batch.to(self.device)
                soil_batch = soil_batch.to(self.device)
                coords_batch = coords_batch.to(self.device)
                past_yields_batch = past_yields_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass - need to create dummy inputs to match expected signature
                n_years, weather_vars, weeks = (
                    weather_batch.shape[1],
                    weather_batch.shape[2],
                    weather_batch.shape[3],
                )
                seq_len = n_years * weeks

                # Create padded_weather in expected format (batch, seq_len, total_weather_vars)
                batch_size_actual = weather_batch.shape[0]
                padded_weather = torch.zeros(batch_size_actual, seq_len, 50).to(
                    self.device
                )  # 50 total weather vars
                weather_indices = [7, 8, 11, 1, 2, 29]

                # Reshape weather to (batch, seq_len, 6) and put in correct indices
                weather_flat = weather_batch.view(
                    batch_size_actual, seq_len, weather_vars
                )
                padded_weather[:, :, weather_indices] = weather_flat

                # Create dummy inputs for compatibility
                year_expanded = torch.zeros(batch_size_actual, seq_len).to(self.device)
                interval = torch.zeros(batch_size_actual, 1).to(self.device)
                weather_feature_mask = torch.ones(batch_size_actual, seq_len, 50).to(
                    self.device
                )

                # Forward pass
                predicted_yield = self.model(
                    padded_weather,
                    coords_batch,
                    year_expanded,
                    interval,
                    weather_feature_mask,
                    soil_batch,
                    past_yields_batch,
                )

                # Compute loss
                loss = self.criterion(predicted_yield.squeeze(), targets_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        else:
            # Regular dataloader (fallback)
            for batch in train_loader:
                # Handle regular batch format if needed
                pass

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Epoch {epoch_idx + 1} - Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self, val_loader):
        """Validate with GNN-RNN specific handling"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            if hasattr(val_loader, "sequences"):
                dataset = val_loader
                batch_size = min(32, len(dataset))

                for i in range(0, len(dataset), batch_size):
                    batch_data = []
                    for j in range(i, min(i + batch_size, len(dataset))):
                        batch_data.append(dataset[j])

                    if not batch_data:
                        continue

                    # Collate batch
                    weather_batch = torch.stack(
                        [torch.FloatTensor(item["weather"]) for item in batch_data]
                    )
                    soil_batch = torch.stack(
                        [torch.FloatTensor(item["soil"]) for item in batch_data]
                    )
                    coords_batch = torch.stack(
                        [torch.FloatTensor(item["coords"]) for item in batch_data]
                    )
                    past_yields_batch = torch.stack(
                        [torch.FloatTensor(item["past_yields"]) for item in batch_data]
                    )
                    targets_batch = torch.FloatTensor(
                        [item["target_yield"] for item in batch_data]
                    )

                    # Move to device
                    weather_batch = weather_batch.to(self.device)
                    soil_batch = soil_batch.to(self.device)
                    coords_batch = coords_batch.to(self.device)
                    past_yields_batch = past_yields_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device)

                    # Forward pass
                    n_years, weather_vars, weeks = (
                        weather_batch.shape[1],
                        weather_batch.shape[2],
                        weather_batch.shape[3],
                    )
                    seq_len = n_years * weeks

                    batch_size_actual = weather_batch.shape[0]
                    padded_weather = torch.zeros(batch_size_actual, seq_len, 50).to(
                        self.device
                    )
                    weather_indices = [7, 8, 11, 1, 2, 29]

                    weather_flat = weather_batch.view(
                        batch_size_actual, seq_len, weather_vars
                    )
                    padded_weather[:, :, weather_indices] = weather_flat

                    year_expanded = torch.zeros(batch_size_actual, seq_len).to(
                        self.device
                    )
                    interval = torch.zeros(batch_size_actual, 1).to(self.device)
                    weather_feature_mask = torch.ones(
                        batch_size_actual, seq_len, 50
                    ).to(self.device)

                    predicted_yield = self.model(
                        padded_weather,
                        coords_batch,
                        year_expanded,
                        interval,
                        weather_feature_mask,
                        soil_batch,
                        past_yields_batch,
                    )

                    # Compute RMSE loss for validation
                    loss = torch.sqrt(
                        self.criterion(predicted_yield.squeeze(), targets_batch)
                    )
                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def compute_train_loss(
        self,
        padded_weather,
        coord_processed,
        year_expanded,
        interval,
        weather_feature_mask,
        practices,
        soil,
        y_past,
        target_yield,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for GNN-RNN model"""
        predicted_yield = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            soil,
            y_past,
        )

        # Compute MSE loss
        loss = self.criterion(predicted_yield.squeeze(), target_yield.squeeze())
        return {"total_loss": loss}

    def compute_validation_loss(
        self,
        padded_weather,
        coord_processed,
        year_expanded,
        interval,
        weather_feature_mask,
        practices,
        soil,
        y_past,
        target_yield,
    ) -> Dict[str, torch.Tensor]:
        """Compute validation loss for GNN-RNN model"""
        with torch.no_grad():
            predicted_yield = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                soil,
                y_past,
            )

        # Return RMSE for validation
        loss = self.criterion(predicted_yield.squeeze(), target_yield.squeeze())
        return {"total_loss": loss**0.5}


def gnnrnn_yield_training_loop(args_dict, use_cropnet: bool):
    """
    GNN-RNN training loop using GNN-specific dataloader and training approach
    """
    logger = logging.getLogger(__name__)

    # Read the dataset
    data_dir = "/Users/adibhasan/Downloads/WeatherModel/data/"  # Adjust path as needed
    crop_df = read_soybean_dataset(data_dir)

    crop_type = args_dict["crop_type"]
    test_years = EXTREME_YEARS[crop_type]

    # Cross-validation setup
    # test_years = (
    #     TEST_YEARS if args_dict.get("test_year") is None else [args_dict["test_year"]]
    # )
    fold_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    for test_year in test_years:
        logger.info(f"Training for test year: {test_year}")

        # Get GNN-specific dataloaders
        train_dataset, test_dataset, nodeloader = get_gnnrnn_dataloaders(
            crop_df=crop_df,
            test_year=test_year,
            n_past_years=args_dict["n_past_years"],
            batch_size=args_dict["batch_size"],
            device=device,
            crop_type=args_dict["crop_type"],
        )

        logger.info(
            f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
        )

        # Create model
        model = GNNRNNYieldModel(
            name=f"gnnrnn_{args_dict['crop_type']}_yield_{test_year}",
            device=device,
            weather_dim=6,
            n_past_years=args_dict["n_past_years"],
        ).to(device)

        # Create optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=args_dict["init_lr"])
        criterion = nn.MSELoss()

        # Create trainer
        trainer = GNNRNNYieldTrainer(
            crop_df=crop_df,
            n_past_years=args_dict["n_past_years"],
            n_train_years=args_dict["n_train_years"],
            beta=args_dict.get("beta", 1e-4),
            use_cropnet=use_cropnet,
            crop_type=args_dict["crop_type"],
            test_year=test_year,
            model=model,
            batch_size=args_dict["batch_size"],
            num_epochs=args_dict["n_epochs"],
            init_lr=args_dict["init_lr"],
        )

        # Set optimizer and criterion manually for our custom training
        trainer.optimizer = optimizer
        trainer.criterion = criterion

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(args_dict["n_epochs"]):
            train_loss = trainer.train_one_epoch(train_dataset, epoch)
            val_loss = trainer.validate(test_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            logger.info(
                f"Epoch {epoch + 1}/{args_dict['n_epochs']} - Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            )

        fold_results.append(best_val_loss)
        logger.info(f"Best validation RMSE for {test_year}: {best_val_loss:.6f}")

    # Return results in expected format
    return {
        "fold_results": fold_results,
        "avg_rmse": np.mean(fold_results),
        "std_rmse": np.std(fold_results),
    }
