import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
import logging

from src.crop_yield.models.gnnrnn_yield_model import GNNRNNYieldModel
from src.crop_yield.dataloader.gnnrnn_dataloader import get_gnnrnn_dataloaders
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

    KEY IMPLEMENTATION DETAILS:
    - Graph nodes represent counties/geographic regions
    - NodeDataLoader samples graph nodes (counties) for each batch
    - Each sampled node is mapped to actual data samples from that county
    - This maintains the geographic structure that's critical for GNN performance
    - PERFORMANCE: All sampling operations run on GPU with pre-computed tensors
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize GNN-specific attributes
        self.nodeloader: Optional[Any] = None
        self.train_node_mapping: Optional[Dict] = None
        self.test_node_mapping: Optional[Dict] = None

    def get_dataloaders(self, shuffle: bool = False) -> Tuple[Any, Any]:
        """Use GNN dataloaders and set up nodeloader"""
        if (
            self.train_loader is not None
            and self.test_loader is not None
            and self.train_node_mapping is not None
            and self.test_node_mapping is not None
        ):
            return self.train_loader, self.test_loader

        (
            train_dataset,
            test_dataset,
            nodeloader,
            train_node_mapping,
            test_node_mapping,
        ) = get_gnnrnn_dataloaders(
            crop_df=self.crop_df,
            test_year=self.test_year,
            n_train_years=self.n_train_years,
            n_past_years=self.n_past_years,
            batch_size=self.batch_size,
            device=self.device,
            crop_type=self.crop_type,
        )

        # Store nodeloader and mappings for use in training/validation
        self.nodeloader = nodeloader
        self.train_node_mapping = train_node_mapping
        self.test_node_mapping = test_node_mapping

        # Pre-compute GPU tensors for fast node-to-sample mapping
        self.train_node_to_samples_gpu = self._create_gpu_mapping(
            train_node_mapping, len(train_dataset)
        )
        self.test_node_to_samples_gpu = self._create_gpu_mapping(
            test_node_mapping, len(test_dataset)
        )
        self.train_loader = train_dataset
        self.test_loader = test_dataset
        return train_dataset, test_dataset

    def _create_gpu_mapping(self, node_mapping: Dict, dataset_len: int) -> torch.Tensor:
        """Create GPU tensor for fast node-to-sample mapping"""
        max_samples_per_node = (
            max(len(samples) for samples in node_mapping.values())
            if node_mapping
            else 1
        )
        num_nodes = max(node_mapping.keys()) + 1 if node_mapping else 1

        # Create mapping tensor: [num_nodes, max_samples_per_node] filled with -1 initially
        mapping_tensor = torch.full(
            (num_nodes, max_samples_per_node), -1, dtype=torch.long, device=self.device
        )

        for node_idx, sample_indices in node_mapping.items():
            for i, sample_idx in enumerate(sample_indices):
                if i < max_samples_per_node:
                    mapping_tensor[node_idx, i] = sample_idx

        return mapping_tensor

    def _get_samples_for_nodes_gpu(
        self, node_indices: torch.Tensor, mapping_tensor: torch.Tensor, dataset_len: int
    ):
        """Pure GPU sampling - no CPU transfers!"""
        batch_size = len(node_indices)
        samples = []

        for node_idx in node_indices:
            node_idx = int(node_idx.item())  # Explicit int conversion
            if node_idx < mapping_tensor.size(0):
                # Get valid sample indices for this node (non-negative)
                node_samples = mapping_tensor[node_idx]
                valid_samples = node_samples[node_samples >= 0]

                if len(valid_samples) > 0:
                    # Random choice on GPU
                    rand_idx = torch.randint(
                        len(valid_samples), (1,), device=self.device
                    )
                    chosen_sample = valid_samples[rand_idx].item()
                    samples.append(chosen_sample)
                else:
                    # Fallback: random sample
                    rand_sample = torch.randint(
                        dataset_len, (1,), device=self.device
                    ).item()
                    samples.append(rand_sample)
            else:
                # Fallback: random sample
                rand_sample = torch.randint(
                    dataset_len, (1,), device=self.device
                ).item()
                samples.append(rand_sample)

        return samples

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

        # KEEP THE GNN SAMPLING - use proper node-to-samples mapping
        assert self.nodeloader is not None, "nodeloader should be set"
        assert self.train_node_mapping is not None, "train_node_mapping should be set"

        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(self.nodeloader):
            # Convert blocks to device
            blocks = [block.to(self.device) for block in blocks]

            # Map in_nodes to actual data samples using proper mapping
            num_in_nodes = len(in_nodes)
            if num_in_nodes == 0:
                continue

            # PURE GPU SAMPLING - NO CPU TRANSFERS!
            sample_indices = self._get_samples_for_nodes_gpu(
                in_nodes, self.train_node_to_samples_gpu, len(train_dataset)
            )
            in_batch_items = [train_dataset[idx] for idx in sample_indices]

            # Convert to tensors - features for ALL in_nodes
            weather = torch.stack(
                [torch.FloatTensor(item["weather"]) for item in in_batch_items]
            ).to(self.device)
            soil = torch.stack(
                [torch.FloatTensor(item["soil"]) for item in in_batch_items]
            ).to(self.device)
            coords = torch.stack(
                [torch.FloatTensor(item["coords"]) for item in in_batch_items]
            ).to(self.device)
            past_yields = torch.stack(
                [torch.FloatTensor(item["past_yields"]) for item in in_batch_items]
            ).to(self.device)

            # Targets only for out_nodes (the nodes we're predicting)
            num_out_nodes = len(out_nodes)
            out_batch_items = in_batch_items[
                :num_out_nodes
            ]  # Take first out_nodes items
            targets = torch.FloatTensor(
                [item["target_yield"] for item in out_batch_items]
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
            # KEEP THE GNN SAMPLING for validation - use proper node-to-samples mapping
            assert self.nodeloader is not None, "nodeloader should be set"
            assert self.test_node_mapping is not None, "test_node_mapping should be set"

            for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(self.nodeloader):
                # Convert blocks to device
                blocks = [block.to(self.device) for block in blocks]

                # Map in_nodes to actual data samples using proper mapping
                num_in_nodes = len(in_nodes)
                if num_in_nodes == 0:
                    continue

                    # PURE GPU SAMPLING - NO CPU TRANSFERS!
                sample_indices = self._get_samples_for_nodes_gpu(
                    in_nodes, self.test_node_to_samples_gpu, len(test_dataset)
                )
                in_batch_items = [test_dataset[idx] for idx in sample_indices]

                # Convert to tensors - features for ALL in_nodes
                weather = torch.stack(
                    [torch.FloatTensor(item["weather"]) for item in in_batch_items]
                ).to(self.device)
                soil = torch.stack(
                    [torch.FloatTensor(item["soil"]) for item in in_batch_items]
                ).to(self.device)
                coords = torch.stack(
                    [torch.FloatTensor(item["coords"]) for item in in_batch_items]
                ).to(self.device)
                past_yields = torch.stack(
                    [torch.FloatTensor(item["past_yields"]) for item in in_batch_items]
                ).to(self.device)

                # Targets only for out_nodes (the nodes we're predicting)
                num_out_nodes = len(out_nodes)
                out_batch_items = in_batch_items[
                    :num_out_nodes
                ]  # Take first out_nodes items
                targets = torch.FloatTensor(
                    [item["target_yield"] for item in out_batch_items]
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
