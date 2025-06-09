import torch
import torch.nn as nn
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

FOLD_IDX = 0

TEST_STATES_MAP = [
    ("south dakota", "iowa"),
    ("nebraska", "minnesota"),
    ("north dakota", "kansas"),
    ("indiana", "missouri"),
    ("illinois", "minnesota"),
    ("nebraska", "iowa"),
]


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
        train_pct: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store yield-specific parameters
        self.crop_df = crop_df
        self.available_states = list(sorted(set(crop_df["State"].values)))
        self.n_past_years = n_past_years
        self.train_pct = train_pct
        self.beta = beta
        self.output_json["model_config"]["beta"] = beta
        # Override criterion for yield prediction
        self.criterion = nn.MSELoss(reduction="mean")

        # Override model directory for yield prediction
        if self.rank == 0:
            self.model_dir = DATA_DIR + "trained_models/crop_yield/"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        global FOLD_IDX
        self.test_states = TEST_STATES_MAP[FOLD_IDX]
        FOLD_IDX += 1
        self.logger.info(f"Testing on states: {','.join(self.test_states)}")

        # Cache for datasets to avoid recreation during cross-validation
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    # =============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (required by BaseTrainer)
    # =============================================================================

    def get_dataloaders(self, shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation - IMPLEMENTATION OF ABSTRACT METHOD."""

        if self.train_loader is not None and self.test_loader is not None:
            return self.train_loader, self.test_loader

        train_loader, test_loader = get_train_test_loaders(
            self.crop_df,
            self.test_states,
            self.n_past_years,
            self.batch_size,
            shuffle,
            num_workers=0 if self.world_size > 1 else 8,
            train_pct=self.train_pct,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
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
        return {"total_loss": loss**0.5}

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

        # Return RMSE for validation since that's standard for comparision
        loss = self.criterion(predicted_yield.squeeze(), target_yield.squeeze())
        return {"total_loss": loss**0.5}

    def _current_beta(self):
        # num_epochs = self.get_num_epochs()
        # current_epoch = self.get_current_epoch()
        # if current_epoch is None:
        #     raise ValueError("Current epoch is not set")
        # beta_multiplier = 0.0 if current_epoch < 10 else 1.0
        # return self.beta * beta_multiplier
        return self.beta


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================


def _create_yield_training_setup(args_dict):
    """
    Helper function to create common training setup for all yield trainers.
    Returns common parameters needed by all yield training loops.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Calculate MLP input dimension
    mlp_input_dim = TOTAL_WEATHER_VARS * 52 * (args_dict["n_past_years"] + 1)

    # Read dataset
    crop_df = read_soybean_dataset(DATA_DIR)

    # Validate cross validation parameter
    cross_validation_k = args_dict["cross_validation_k"]
    if cross_validation_k is None or cross_validation_k <= 1:
        raise ValueError("Cross validation k must be greater than 1")

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "mlp_input_dim": mlp_input_dim,
        "crop_df": crop_df,
        "cross_validation_k": cross_validation_k,
        "beta": args_dict["beta"],
    }


def _run_yield_cross_validation(
    setup_params,
    model_class,
    trainer_class,
    model_name,
    args_dict,
    extra_trainer_kwargs=None,
    extra_model_kwargs=None,
):
    """
    Helper function to run cross-validation for yield prediction models.

    Args:
        setup_params: Dictionary from _create_yield_training_setup()
        model_class: Model class to instantiate
        trainer_class: Trainer class to use
        model_name: Name for the model
        args_dict: Original arguments dictionary
        extra_trainer_kwargs: Additional trainer-specific kwargs (optional)
        extra_model_kwargs: Additional model-specific kwargs (optional)
    """
    model_kwargs = {
        "name": model_name,
        "mlp_input_dim": setup_params["mlp_input_dim"],
        "weather_dim": TOTAL_WEATHER_VARS,
        "output_dim": TOTAL_WEATHER_VARS,
        "device": setup_params["device"],
        **args_dict["model_size_params"],
    }

    # Add any extra model-specific kwargs
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    trainer_kwargs = {
        "crop_df": setup_params["crop_df"],
        "n_past_years": args_dict["n_past_years"],
        "train_pct": args_dict["train_pct"],
        "batch_size": args_dict["batch_size"],
        "num_epochs": args_dict["n_epochs"],
        "init_lr": args_dict["init_lr"],
        "num_warmup_epochs": args_dict["n_warmup_epochs"],
        "decay_factor": args_dict["decay_factor"],
        "pretrained_model_path": args_dict["pretrained_model_path"],
        "resume_from_checkpoint": args_dict.get("resume_from_checkpoint"),
        "rank": setup_params["rank"],
        "world_size": setup_params["world_size"],
        "local_rank": setup_params["local_rank"],
    }

    # Add any extra trainer-specific kwargs
    if extra_trainer_kwargs:
        trainer_kwargs.update(extra_trainer_kwargs)

    cross_validator = CrossValidator(
        model_class=model_class,
        model_kwargs=model_kwargs,
        trainer_class=trainer_class,
        trainer_kwargs=trainer_kwargs,
        k_folds=setup_params["cross_validation_k"],
    )

    return cross_validator.run_cross_validation(
        use_optimal_lr=args_dict["use_optimal_lr"]
    )


def weatherbert_yield_training_loop(args_dict):
    """
    BERT training loop using the WeatherBertYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherBERTYieldModel,
        trainer_class=WeatherBERTYieldTrainer,
        model_name="weatherbert_yield",
        args_dict=args_dict,
    )
