import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict
from src.crop_yield.trainers.weatherbert_yield_trainer import WeatherBERTYieldTrainer
import logging
from src.crop_yield.models.weatherformer_yield_model import WeatherFormerYieldModel
from src.crop_yield.dataloader.yield_dataloader import read_soybean_dataset
from src.utils.constants import DATA_DIR, TOTAL_WEATHER_VARS
from src.base_trainer.cross_validator import CrossValidator


class WeatherFormerYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherFormer-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a variational loss function
    that accounts for the probabilistic nature of WeatherFormer's (mu, sigma) outputs.

    The loss function includes:
    1. Reconstruction term: (y - μ_θ(z))²
    2. KL divergence term: σ_y² ∑(μ²_φ,d + σ²_φ,d - log σ_φ,d)
    """

    def __init__(self, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss()
        self.sigma_y_squared = beta

        # override the loss collection
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
        }

    def get_model_name(self) -> str:
        return "weatherformer_yield"

    def _compute_variational_loss_components(
        self,
        mu_x: torch.Tensor,
        sigma_squared_x: torch.Tensor,
        yield_pred: torch.Tensor,
        target_yield: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the variational loss components for WeatherFormer yield prediction.

        Args:
            yield_pred: Predicted yield values
            mu_x: Mean of weather representations
            sigma_squared_x: Variance of weather representations
            target_yield: Ground truth yield values

        Returns:
            Dictionary containing all loss components
        """
        # 1. Reconstruction term: (y_j - μ_θ(z_j))²
        reconstruction_loss = self.criterion(
            yield_pred.squeeze(), target_yield.squeeze()
        )
        # 3. KL divergence term: σ_y² ∑(μ²_φ,d + σ²_φ,d - log σ_φ,d)
        # Sum over all dimensions d
        mu_squared = torch.sum(mu_x**2, dim=-1)  # Sum over weather dimensions
        sigma_squared = torch.sum(
            sigma_squared_x, dim=-1
        )  # Sum over weather dimensions
        log_sigma_squared = torch.sum(
            torch.log(sigma_squared_x), dim=-1
        )  # Sum over weather dimensions

        kl_term = self.sigma_y_squared * torch.sum(
            mu_squared + sigma_squared - log_sigma_squared,
            dim=-1,  # Sum over sequence length
        )
        kl_term = torch.mean(kl_term)  # Average over batch

        # Total loss
        total_loss = reconstruction_loss + kl_term

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_term": kl_term,
        }

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
        """Compute variational training loss for WeatherFormer yield prediction."""

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherFormer model
        # Returns (yield_pred, mu_x, sigma_x)
        yield_pred, mu_x, sigma_squared_x = self.model(input_data)

        # Compute all loss components using the helper method
        return self._compute_variational_loss_components(
            mu_x, sigma_squared_x, yield_pred, target_yield
        )

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
        """Compute variational validation loss for WeatherFormer yield prediction."""

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherFormer model (no gradient computation)
        with torch.no_grad():
            yield_pred, mu_x, sigma_squared_x = self.model(input_data)

        # Compute all loss components using the helper method
        return self._compute_variational_loss_components(
            mu_x, sigma_squared_x, yield_pred, target_yield
        )


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherformer_yield_training_loop(args_dict):
    """
    WeatherFormer training loop using the WeatherFormerYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    mlp_input_dim = TOTAL_WEATHER_VARS * 52 * (args_dict["n_past_years"] + 1)

    # Initialize WeatherFormer model
    model = WeatherFormerYieldModel(
        name="weatherformer_yield",
        mlp_input_dim=mlp_input_dim,
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
        "beta": args_dict["beta"],  # WeatherFormer-specific parameter
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }

    cross_validator = CrossValidator(
        trainer_class=WeatherFormerYieldTrainer,
        trainer_kwargs=trainer_kwargs,
        k_folds=cross_validation_k,
    )

    return cross_validator.run_cross_validation()
