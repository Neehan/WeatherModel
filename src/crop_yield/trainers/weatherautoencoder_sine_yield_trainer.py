import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Dict, Tuple
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherautoencoder_sine_yield_model import (
    WeatherAutoencoderSineYieldModel,
)


class WeatherAutoencoderSineYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherAutoencoderSine-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a sinusoidal loss function
    that accounts for the sinusoidal prior nature of WeatherAutoencoderSine's outputs.

    The loss function includes:
    1. Reconstruction term: (y - μ_θ(z))²
    2. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is the sinusoidal prior
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction="mean")
        # override the loss collection to match expected keys
        if self.rank == 0:
            self.output_json["losses"] = {
                "train": {
                    "total_loss": [],
                    "reconstruction": [],
                    "log_variance": [],
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],  # just MSE
                },
            }

    def _compute_sine_kl_divergence(
        self,
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior means
        var_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior variances
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence between posterior q(z|x) and sinusoidal prior p(z).

        KL(q(z|x) || p(z)) where both are normal distributions:
        - q(z|x) ~ N(mu_x, var_x)
        - p(z) ~ N(mu_p, var_p) where mu_p = A_p * sin(theta_p * z)

        KL(N(μ₁, σ₁²) || N(μ₂, σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁ - μ₂)²)/(2σ₂²) - 1/2
        """
        # Compute KL divergence between two normal distributions
        # KL(N(mu_x, var_x) || N(mu_p, var_p))
        log_variance_x = torch.log(var_x)
        log_variance_p = torch.log(var_p)

        kl_divergence = 0.5 * torch.sum(
            log_variance_p
            - log_variance_x  # log(σ_p/σ_x)
            + var_x / var_p  # σ_x²/σ_p²
            + (mu_x - mu_p) ** 2 / var_p  # (μ_x - μ_p)²/σ_p²
            - 1,  # -1
            dim=(1, 2),  # sum over seq_len and n_features
        )  # [batch_size]

        return kl_divergence.mean(), log_variance_x.mean()  # average over batch

    def _compute_sine_variational_loss_components(
        self,
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior means
        var_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior variances
        yield_pred: torch.Tensor,  # [batch_size, 1]
        target_yield: torch.Tensor,  # [batch_size, 1]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sinusoidal variational loss components for WeatherAutoencoderSine yield prediction.

        Implements the sinusoidal ELBO loss:
        L_yield = ||y_j - μ_θ(z_j)||² + β · KL(q(z_j|x_j) || p(z_j))

        where p(z_j) is the sinusoidal prior: N(A_p * sin(theta_p * z), var_p).

        Args:
            z: Sampled latent representations [batch_size, seq_len, n_features]
            mu_x: Posterior means [batch_size, seq_len, n_features]
            var_x: Posterior variances [batch_size, seq_len, n_features]
            mu_p: Sinusoidal prior means [batch_size, seq_len, n_features]
            var_p: Sinusoidal prior variances [batch_size, seq_len, n_features]
            yield_pred: Predicted yield values [batch_size, 1]
            target_yield: Ground truth yield values [batch_size, 1]

        Returns:
            Dictionary containing all loss components
        """
        # 1. Reconstruction term: mean over batch of (y_j - μ_θ(z_j))²
        reconstruction_loss = self.criterion(
            yield_pred.squeeze(), target_yield.squeeze()
        )

        # 2. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is sinusoidal prior
        kl_term, log_variance = self._compute_sine_kl_divergence(
            z, mu_x, var_x, mu_p, var_p
        )
        # Average over batch and multiply by β
        beta = self._current_beta()
        kl_term = beta * kl_term

        # Total loss: sum of both terms
        total_loss = reconstruction_loss + kl_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_loss,
            "kl_term": kl_term,
            "log_variance": log_variance,
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
        """Compute sinusoidal variational training loss for WeatherAutoencoderSine yield prediction."""

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherAutoencoderSine model
        # Returns (yield_pred, z, mu_x, var_x, mu_p, var_p)
        yield_pred, z, mu_x, var_x, mu_p, var_p = self.model(input_data)

        # Compute all loss components using the helper method
        return self._compute_sine_variational_loss_components(
            z, mu_x, var_x, mu_p, var_p, yield_pred, target_yield
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
        """Compute sinusoidal variational validation loss for WeatherAutoencoderSine yield prediction."""

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherAutoencoderSine model (no gradient computation)
        with torch.no_grad():
            yield_pred, z, mu_x, var_x, mu_p, var_p = self.model(input_data)

        # Compute all loss components using the helper method
        components = self._compute_sine_variational_loss_components(
            z, mu_x, var_x, mu_p, var_p, yield_pred, target_yield
        )
        # only return the reconstruction (RMSE) loss for validation
        return {"total_loss": components["reconstruction"] ** 0.5}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherautoencoder_sine_yield_training_loop(args_dict):
    """
    WeatherAutoencoderSine training loop using the WeatherAutoencoderSineYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    # WeatherAutoencoderSine-specific trainer kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    # WeatherAutoencoderSine-specific model kwargs - no mixture components needed
    extra_model_kwargs = {}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherAutoencoderSineYieldModel,
        trainer_class=WeatherAutoencoderSineYieldTrainer,
        model_name="weatherautoencoder_sine_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
        extra_model_kwargs=extra_model_kwargs,
    )
