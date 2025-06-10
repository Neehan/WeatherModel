import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherformer_yield_model import WeatherFormerYieldModel


class WeatherFormerYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherFormer-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a variational loss function
    that accounts for the probabilistic nature of WeatherFormer's (mu, sigma) outputs.

    The loss function includes:
    1. Reconstruction term: (y - μ_θ(z))²
    2. KL divergence term: β * ∑(μ²_φ,d + sigma²_φ,d - log sigma²_φ,d)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction="mean")
        # override the loss collection to match expected k
        if self.rank == 0:
            self.output_json["losses"] = {
                "train": {
                    "total_loss": [],
                    "reconstruction": [],
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],  # just MSE
                },
            }

    def _compute_variational_loss_components(
        self,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        yield_pred: torch.Tensor,
        target_yield: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the variational loss components for WeatherFormer yield prediction.

        Implements the exact formula from the paper:
        L_yield = ∑_{j=1}^{N_y} [(y_j - μ_θ(z_j))² + β ∑_{d=1}^D (μ_{φ,d}²(x_j) + σ_{φ,d}²(x_j) - log σ_{φ,d}²(x_j))]

        Args:
            yield_pred: Predicted yield values [batch_size, 1]
            mu_x: Mean of weather representations [batch_size, seq_len, weather_dim]
            var_x: Variance of weather representations [batch_size, seq_len, weather_dim]
            target_yield: Ground truth yield values [batch_size, 1]

        Returns:
            Dictionary containing all loss components
        """
        # 1. Reconstruction term: mean over batch of (y_j - μ_θ(z_j))²
        # Use MSELoss for proper averaging
        reconstruction_loss = self.criterion(
            yield_pred.squeeze(), target_yield.squeeze()
        )

        # 2. KL divergence term: β * mean over batch of ∑_{d=1}^D (μ_{φ,d}²(x_j) + σ_{φ,d}²(x_j) - log σ_{φ,d}²(x_j))
        # First, compute the KL term for each sample j and dimension d
        kl_per_sample_per_dim = mu_x**2 + var_x - torch.log(var_x)

        # Sum over all dimensions D (both sequence length and weather dimensions)
        kl_per_sample = torch.sum(
            kl_per_sample_per_dim.view(mu_x.size(0), -1), dim=1
        )  # [batch_size]

        # Average over batch and multiply by β
        beta = self._current_beta()
        kl_term = beta * torch.mean(kl_per_sample)

        # Total loss: sum of both terms
        total_loss = reconstruction_loss + kl_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_loss,
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

        # Forward pass through WeatherFormer model
        # Returns (yield_pred, mu_x, sigma_x)
        yield_pred, mu_x, var_x = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            practices,
            soil,
            y_past,
        )

        # Compute all loss components using the helper method
        return self._compute_variational_loss_components(
            mu_x, var_x, yield_pred, target_yield
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

        # Forward pass through WeatherFormer model (no gradient computation)
        with torch.no_grad():
            yield_pred, mu_x, var_x = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                practices,
                soil,
                y_past,
            )

        # Compute all loss components using the helper method
        components = self._compute_variational_loss_components(
            mu_x, var_x, yield_pred, target_yield
        )
        # only return the reconstruction (RMSE) loss for validation
        return {"total_loss": components["reconstruction"] ** 0.5}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherformer_yield_training_loop(args_dict):
    """
    WeatherFormer training loop using the WeatherFormerYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    # WeatherFormer-specific trainer kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherFormerYieldModel,
        trainer_class=WeatherFormerYieldTrainer,
        model_name="weatherformer_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
    )
