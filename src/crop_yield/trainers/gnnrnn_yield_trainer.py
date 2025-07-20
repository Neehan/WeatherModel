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
    DATA_DIR,
)


class GNNRNNYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for GNN-RNN crop yield prediction models.
    Uses standard WeatherBERT training with minimal GNN-specific overrides.
    """

    def get_dataloaders(self, shuffle: bool = False):
        """Use GNN dataloaders instead of regular yield dataloaders"""
        if self.train_loader is not None and self.test_loader is not None:
            return self.train_loader, self.test_loader

        train_dataset, test_dataset, nodeloader = get_gnnrnn_dataloaders(
            crop_df=self.crop_df,
            test_year=self.test_year,
            n_past_years=self.n_past_years,
            batch_size=self.batch_size,
            device=self.device,
            crop_type=self.crop_type,
        )

        self.train_loader = train_dataset
        self.test_loader = test_dataset
        return train_dataset, test_dataset

    def _train_epoch(self, train_loader):
        """Handle GNN dataset format"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        dataset = train_loader
        for i in range(0, len(dataset), self.batch_size):
            batch_items = [
                dataset[j] for j in range(i, min(i + self.batch_size, len(dataset)))
            ]
            if not batch_items:
                continue

            # Convert to tensors and move to device - NO CONVERSION NEEDED!
            weather = torch.stack(
                [torch.FloatTensor(item["weather"]) for item in batch_items]
            ).to(self.device)
            soil = torch.stack(
                [torch.FloatTensor(item["soil"]) for item in batch_items]
            ).to(self.device)
            coords = torch.stack(
                [torch.FloatTensor(item["coords"]) for item in batch_items]
            ).to(self.device)
            past_yields = torch.stack(
                [torch.FloatTensor(item["past_yields"]) for item in batch_items]
            ).to(self.device)
            targets = torch.FloatTensor(
                [item["target_yield"] for item in batch_items]
            ).to(self.device)

            # Forward pass - direct GNN format
            self.optimizer.zero_grad()
            predicted_yield = self.model(weather, soil, coords, past_yields)

            loss = self.criterion(predicted_yield.squeeze(), targets.squeeze())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, val_loader):
        """Handle GNN dataset format"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        dataset = val_loader
        with torch.no_grad():
            for i in range(0, len(dataset), self.batch_size):
                batch_items = [
                    dataset[j] for j in range(i, min(i + self.batch_size, len(dataset)))
                ]
                if not batch_items:
                    continue

                weather = torch.stack(
                    [torch.FloatTensor(item["weather"]) for item in batch_items]
                ).to(self.device)
                soil = torch.stack(
                    [torch.FloatTensor(item["soil"]) for item in batch_items]
                ).to(self.device)
                coords = torch.stack(
                    [torch.FloatTensor(item["coords"]) for item in batch_items]
                ).to(self.device)
                past_yields = torch.stack(
                    [torch.FloatTensor(item["past_yields"]) for item in batch_items]
                ).to(self.device)
                targets = torch.FloatTensor(
                    [item["target_yield"] for item in batch_items]
                ).to(self.device)

                predicted_yield = self.model(weather, soil, coords, past_yields)

                loss = self.criterion(predicted_yield.squeeze(), targets.squeeze())
                total_loss += loss.item() ** 0.5  # RMSE for validation
                num_batches += 1

        return total_loss / max(num_batches, 1)


def gnnrnn_yield_training_loop(args_dict, use_cropnet: bool):
    """
    GNN-RNN training loop using standard cross-validation approach like other models
    """
    from src.crop_yield.trainers.weatherbert_yield_trainer import (
        _create_yield_training_setup,
        _run_yield_cross_validation,
    )
    from src.crop_yield.models.gnnrnn_yield_model import GNNRNNYieldModel

    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=GNNRNNYieldModel,
        trainer_class=GNNRNNYieldTrainer,
        model_name=f"gnnrnn_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
    )
