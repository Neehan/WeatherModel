import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
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
    Uses BaseTrainer infrastructure with GNN-specific data handling following original paper.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize nodeloader as None - will be set in get_dataloaders
        self.nodeloader: Optional[Any] = None

    def get_dataloaders(self, shuffle: bool = False) -> Tuple:
        """Use GNN dataloaders and set up nodeloader"""
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

        # Store nodeloader for use in training/validation
        self.nodeloader = nodeloader
        self.train_loader = train_dataset
        self.test_loader = test_dataset
        return train_dataset, test_dataset

    def compute_train_loss(self, *args) -> Dict[str, torch.Tensor]:
        """Not used - we override _train_epoch directly following original paper pattern"""
        raise NotImplementedError(
            "GNN-RNN uses custom training loop - this method is not used"
        )

    def compute_validation_loss(self, *args) -> Dict[str, torch.Tensor]:
        """Not used - we override _validate_epoch directly following original paper pattern"""
        raise NotImplementedError(
            "GNN-RNN uses custom validation loop - this method is not used"
        )

    def _train_epoch(self, train_dataset):
        """Train epoch following original paper's pattern"""
        self.model.train()
        total_loss_dict = self._initialize_loss_dict("train")

        if self.rank == 0:
            self.logger.info("Started training epoch.")

        num_batches = 0

        # Follow original paper pattern exactly: nodeloader samples graph nodes
        assert self.nodeloader is not None, "nodeloader should be set"

        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(self.nodeloader):
            # Convert blocks to int and move to device (following original paper)
            blocks = [block.int().to(self.device) for block in blocks]

            # Sample data for the output nodes (like original paper's load_subtensor)
            batch_size_actual = len(out_nodes)
            if batch_size_actual == 0:
                continue

            # Sample from our dataset - get random samples equal to out_nodes size
            sampled_indices = torch.randperm(len(train_dataset))[:batch_size_actual]
            batch_items = [train_dataset[i] for i in sampled_indices]

            # Convert to tensors following original paper format
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

            # Forward pass with blocks (following original SAGE_RNN pattern)
            self.optimizer.zero_grad()
            predicted_yield = self.model(weather, soil, coords, past_yields, blocks)

            # MSE loss for training
            loss = self.criterion(predicted_yield.squeeze(), targets.squeeze())
            loss.backward()
            self.optimizer.step()

            # Track loss
            loss_dict = {"total_loss": loss}
            self._accumulate_losses(total_loss_dict, loss_dict)
            num_batches += 1

        self.scheduler.step()
        self._sync_distributed_training()

        avg_loss_dict = self._average_losses(total_loss_dict, num_batches)
        self._update_output_json_losses("train", avg_loss_dict)

        return avg_loss_dict["total_loss"]

    def _validate_epoch(self, test_dataset):
        """Validate epoch following original paper's pattern"""
        self.model.eval()
        total_loss_dict = self._initialize_loss_dict("val")

        if self.rank == 0:
            self.logger.info("Started validation epoch.")

        num_batches = 0

        with torch.no_grad():
            # Follow original paper pattern exactly: nodeloader samples graph nodes
            assert self.nodeloader is not None, "nodeloader should be set"

            for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(self.nodeloader):
                # Convert blocks to int and move to device (following original paper)
                blocks = [block.int().to(self.device) for block in blocks]

                # Sample data for the output nodes (like original paper's load_subtensor)
                batch_size_actual = len(out_nodes)
                if batch_size_actual == 0:
                    continue

                # Sample from our dataset - get random samples equal to out_nodes size
                sampled_indices = torch.randperm(len(test_dataset))[:batch_size_actual]
                batch_items = [test_dataset[i] for i in sampled_indices]

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

                predicted_yield = self.model(weather, soil, coords, past_yields, blocks)

                # RMSE loss for validation (following your convention)
                mse_loss = self.criterion(predicted_yield.squeeze(), targets.squeeze())
                rmse_loss = mse_loss**0.5

                loss_dict = {"total_loss": rmse_loss}
                self._accumulate_losses(total_loss_dict, loss_dict)
                num_batches += 1

        self._sync_distributed_training()

        avg_loss_dict = self._average_losses(total_loss_dict, num_batches)
        self._update_output_json_losses("val", avg_loss_dict)

        return avg_loss_dict["total_loss"]


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
