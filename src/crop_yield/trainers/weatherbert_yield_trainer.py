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
    read_usa_dataset,
    read_non_us_dataset,
)
from src.crop_yield.dataloader.cropnet_dataloader import (
    get_cropnet_train_test_loaders,
    read_cropnet_dataset,
)
from src.base_trainer.cross_validator import CrossValidator
import os

# Test years for 5-fold cross validation
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]
FOLD_IDX = 0

EXTREME_YEARS = {
    "usa": {
        "corn": [2002, 2004, 2009, 2012, 2014],
        "soybean": [2003, 2004, 2009, 2012, 2016],
    },
    "argentina": {
        "corn": [2004, 2005, 2007, 2009, 2015],
        "soybean": [2003, 2006, 2007, 2009, 2015],
        "wheat": [2002, 2003, 2005, 2009, 2011],
        "sunflower": [2002, 2007, 2008, 2009, 2011],
    },
    "brazil": {
        "sugarcane": [2002, 2003, 2008, 2012, 2017],
        "wheat": [2001, 2003, 2010, 2015, 2016],
        "cotton": [2004, 2008, 2013, 2017, 2018],
    },
}


def _reset_fold_index():
    """Reset the global fold index for a new cross-validation run."""
    global FOLD_IDX
    FOLD_IDX = 0


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
        country: str,
        n_past_years: int,
        n_train_years: int,
        beta: float,
        use_cropnet: bool,
        crop_type: str,
        test_year: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store yield-specific parameters
        self.crop_df = crop_df
        self.country = country
        self.n_past_years = n_past_years
        self.n_train_years = n_train_years
        self.beta = beta
        self.use_cropnet = use_cropnet
        self.crop_type = crop_type
        self.output_json["model_config"]["beta"] = beta
        # Override criterion for yield prediction
        self.criterion = nn.MSELoss(reduction="mean")

        # extreme years
        test_years = EXTREME_YEARS.get(country, {}).get(crop_type)
        if test_years is None:
            raise ValueError(f"No extreme years found for {crop_type} in {country}.")
        # test_years = TEST_YEARS

        # Override model directory for yield prediction
        if self.rank == 0:
            self.model_dir = DATA_DIR + "trained_models/crop_yield/"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        # For CropNet, use k=1 fold with test year 2021
        if self.use_cropnet:
            self.test_year = 2021
            self.logger.info(f"CropNet mode - Testing on year: {self.test_year}")
        else:
            if test_year is not None:
                # Single test year mode
                self.test_year = test_year
                self.logger.info(
                    f"Single test year mode - Testing on year: {self.test_year}"
                )
            else:
                # Cross-validation mode
                global FOLD_IDX
                if FOLD_IDX >= len(test_years):
                    raise ValueError(
                        f"FOLD_IDX ({FOLD_IDX}) exceeds TEST_YEARS length ({len(test_years)}). Call _reset_fold_index() before starting new cross-validation."
                    )
                self.test_year = test_years[FOLD_IDX]
                FOLD_IDX += 1
                self.logger.info(
                    f"Cross-validation mode - Testing on year: {self.test_year}"
                )

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

        if self.use_cropnet:
            if self.crop_type is None:
                raise ValueError("crop_type must be specified when using CropNet")
            train_loader, test_loader = get_cropnet_train_test_loaders(
                self.crop_df,
                self.crop_type,
                self.n_train_years,
                self.test_year,
                self.n_past_years,
                self.batch_size,
                shuffle,
                num_workers=0 if self.world_size > 1 else 8,
            )
        else:
            train_loader, test_loader = get_train_test_loaders(
                self.crop_df,
                self.n_train_years,
                self.test_year,
                self.n_past_years,
                self.batch_size,
                shuffle,
                num_workers=0 if self.world_size > 1 else 8,
                crop_type=self.crop_type,
                country=self.country,
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
        # Forward pass through the model
        predicted_yield = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
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
        """Compute validation loss for a batch - IMPLEMENTATION OF ABSTRACT METHOD."""

        # Forward pass through the model (no gradient computation needed for validation)
        with torch.no_grad():
            predicted_yield = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
            )

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


def _create_yield_training_setup(args_dict, use_cropnet: bool):
    """
    Helper function to create common training setup for all yield trainers.
    Returns common parameters needed by all yield training loops.

    Args:
        args_dict: Arguments dictionary
        use_cropnet: Whether to use CropNet training
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    # Force CPU for Mac MPS to avoid DGL compatibility issues
    if torch.backends.mps.is_available():
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA from DGL
        device = torch.device("cpu")  # Force CPU on Mac to avoid DGL MPS issues
        print("Detected MPS device, forcing CPU for DGL compatibility")
    else:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    if use_cropnet:
        # Use provided CropNet DataFrame
        crop_df = read_cropnet_dataset(DATA_DIR)
        # For CropNet, use k=1 fold (test on 2021)
        cross_validation_k = 1
    else:
        # Read dataset based on country
        country = args_dict["country"]
        if country == "argentina":
            crop_df = read_non_us_dataset(DATA_DIR)
        else:  # usa
            crop_df = read_usa_dataset(DATA_DIR)

        # Check if specific test year is provided
        if args_dict.get("test_year") is not None:
            # Single test year mode
            cross_validation_k = 1
        else:
            # Use fixed k-fold cross validation with test years [2014, 2015, 2016, 2017, 2018]
            cross_validation_k = len(TEST_YEARS)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "crop_df": crop_df,
        "cross_validation_k": cross_validation_k,
        "beta": args_dict["beta"],
        "use_cropnet": use_cropnet,
        "test_year": args_dict.get("test_year"),
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
    # Reset fold index before starting new cross-validation (only for multi-fold cross-validation)
    if not setup_params["use_cropnet"] and setup_params["test_year"] is None:
        _reset_fold_index()

    model_kwargs = {
        "name": model_name,
        "device": setup_params["device"],
        "weather_dim": TOTAL_WEATHER_VARS,
        "n_past_years": args_dict["n_past_years"],
        **args_dict["model_size_params"],
    }

    # Add any extra model-specific kwargs
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    trainer_kwargs = {
        "crop_df": setup_params["crop_df"],
        "country": args_dict["country"],
        "n_past_years": args_dict["n_past_years"],
        "n_train_years": args_dict["n_train_years"],
        "beta": args_dict["beta"],
        "use_cropnet": setup_params["use_cropnet"],
        "crop_type": args_dict["crop_type"],
        "test_year": setup_params["test_year"],
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


def weatherbert_yield_training_loop(args_dict, use_cropnet: bool):
    """
    BERT training loop using the WeatherBertYieldTrainer class.
    Initializes the model internally and handles all training.

    Args:
        args_dict: Arguments dictionary
        cropnet_df: Optional CropNet DataFrame for CropNet training
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherBERTYieldModel,
        trainer_class=WeatherBERTYieldTrainer,
        model_name=f"weatherbert_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
    )
