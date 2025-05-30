import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import pandas as pd
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
from src.base_trainer.base_trainer import BaseTrainer
from src.utils.constants import DATA_DIR, TOTAL_WEATHER_VARS
from src.crop_yield.models.weatherbert_yield_model import WeatherBERTYieldModel
from src.crop_yield.dataloader.yield_dataloader import (
    get_train_test_loaders,
    read_soybean_dataset,
)
from src.base_trainer.cross_validator import CrossValidator
import os


class WeatherBERTYieldTrainer(BaseTrainer):
    """
    Trainer class for crop yield prediction models.

    PUBLIC API METHODS (for users):
        - train(): Inherited from BaseTrainer - main training entry point
        - save_checkpoint(): Inherited from BaseTrainer
        - load_checkpoint(): Inherited from BaseTrainer

    ABSTRACT METHOD IMPLEMENTATIONS (required by BaseTrainer):
        - get_dataloaders(): Get train/validation data loaders
        - compute_train_loss(): Compute training loss for a batch
        - compute_validation_loss(): Compute validation loss for a batch
    """

    def __init__(
        self,
        crop_df: pd.DataFrame,
        n_past_years: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store yield-specific parameters
        self.crop_df = crop_df
        self.available_states = set(crop_df["State"].values)
        self.n_past_years = n_past_years

        # Override criterion for yield prediction
        self.criterion = nn.MSELoss()

        # Override model directory for yield prediction
        if self.rank == 0:
            self.model_dir = DATA_DIR + "trained_models/crop_yield/"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    # =============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (required by BaseTrainer)
    # =============================================================================

    def get_dataloaders(
        self, shuffle: bool = True, cross_validation_k: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation - IMPLEMENTATION OF ABSTRACT METHOD."""
        if cross_validation_k is None:
            test_states = ["south dakota", "iowa"]
        else:
            test_states = np.random.choice(
                np.array(sorted(list(self.available_states))),
                size=2,
                replace=False,
            ).tolist()
        self.logger.info(f"Testing on states: {','.join(test_states)}")

        train_loader, test_loader = get_train_test_loaders(
            self.crop_df,
            test_states,
            self.n_past_years,
            self.batch_size,
            shuffle,
        )
        return train_loader, test_loader

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
        """Compute training loss for a batch - IMPLEMENTATION OF ABSTRACT METHOD."""

        # Prepare input data for the model (first 5 elements are what the model expects)
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through the model
        predicted_yield = self.model(input_data)

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
        """Compute validation loss for a batch - IMPLEMENTATION OF ABSTRACT METHOD."""
        # Prepare input data for the model (first 5 elements are what the model expects)
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through the model (no gradient computation needed for validation)
        with torch.no_grad():
            predicted_yield = self.model(input_data)

        loss = F.mse_loss(
            predicted_yield.squeeze(), target_yield.squeeze(), reduction="mean"
        )
        return {"total_loss": loss}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherbert_yield_training_loop(args_dict):
    """
    BERT training loop using the WeatherBertYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize WeatherBERT model
    model = WeatherBERTYieldModel(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    crop_df = read_soybean_dataset(DATA_DIR)

    # Check if cross validation should be used
    cross_validation_k = args_dict["cross_validation_k"]
    if cross_validation_k is None or cross_validation_k <= 1:
        raise ValueError("Cross validation k must be greater than 1")

    # Use CrossValidator for k-fold cross validation
    trainer_kwargs = {
        "model": model,
        "crop_df": crop_df,
        "n_past_years": args_dict["n_past_years"],
        "batch_size": args_dict["batch_size"],
        "num_epochs": args_dict["n_epochs"],
        "init_lr": args_dict["init_lr"],
        "num_warmup_epochs": args_dict["n_warmup_epochs"],
        "decay_factor": args_dict["decay_factor"],
        "pretrained_model_path": args_dict["pretrained_model_path"],
        "masking_prob": args_dict["masking_prob"],
        "masking_n_features": args_dict["masking_n_features"],
        "resume_from_checkpoint": args_dict.get("resume_from_checkpoint"),
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }

    cross_validator = CrossValidator(
        trainer_class=WeatherBERTYieldTrainer,
        trainer_kwargs=trainer_kwargs,
        k_folds=cross_validation_k,
    )

    return cross_validator.run_cross_validation()
