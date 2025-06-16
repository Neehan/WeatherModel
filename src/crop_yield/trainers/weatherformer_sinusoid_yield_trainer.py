import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Dict, Tuple
from src.utils.constants import DRY_RUN
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherformer_sinusoid_yield_model import (
    WeatherFormerSinusoidYieldModel,
)


class WeatherFormerSinusoidYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherFormerSinusoid-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a sinusoidal variational loss function
    that accounts for the sinusoidal prior nature of WeatherFormerSinusoid's outputs.

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
                    "yield": [],
                    "reconstruction": [],
                    "log_variance": [],
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],  # just MSE
                },
            }

    def _masked_mean(
        self, tensor: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]
    ):
        """Mean over `dim`, ignoring False in `mask`."""
        masked = tensor * mask
        return masked.sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1))

    def _compute_sinusoidal_kl_divergence(
        self,
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior mean
        var_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior variance
        weather_feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - weather feature mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence between posterior q(z|x) and sinusoidal prior p(z) for masked features only.

        KL(q(z|x) || p(z)) = 0.5 * [log(var_p/var_x) + var_x/var_p + (mu_x - mu_p)^2/var_p - 1]

        where:
        - q(z|x) is a diagonal Gaussian N(mu_x, var_x)
        - p(z) is a diagonal Gaussian N(mu_p, var_p) with sinusoidal mean
        """
        # Compute KL divergence for each dimension
        kl_per_dim = 0.5 * (
            torch.log(var_p / var_x) + var_x / var_p + (mu_x - mu_p) ** 2 / var_p - 1.0
        )

        # Apply feature mask to only consider masked features
        kl_masked = kl_per_dim * weather_feature_mask

        # Sum over masked features and average over batch
        kl_divergence = self._masked_mean(
            kl_masked, weather_feature_mask, dim=(1, 2)
        ).mean()

        # Compute log variance for monitoring
        log_variance = self._masked_mean(
            torch.log(var_x), weather_feature_mask, dim=(1, 2)
        ).mean()

        return kl_divergence, log_variance

    def _compute_sinusoidal_variational_loss_components(
        self,
        weather: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior mean
        var_p: torch.Tensor,  # [batch_size, seq_len, n_features] - sinusoidal prior variance
        yield_pred: torch.Tensor,  # [batch_size, 1]
        target_yield: torch.Tensor,  # [batch_size, 1]
        weather_feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - mask of predicted positions
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sinusoidal variational loss components for WeatherFormerSinusoid yield prediction.

        Implements the sinusoidal ELBO loss:
        L_yield = ||y_j - μ_θ(z_j)||² + β · KL(q(z_j|x_j) || p(z_j))

        where p(z_j) is the sinusoidal prior and KL divergence is computed only on masked features.

        Args:
            z: Sampled latent representations [batch_size, seq_len, n_features]
            mu_x: Posterior means [batch_size, seq_len, n_features]
            var_x: Posterior variances [batch_size, seq_len, n_features]
            mu_p: Sinusoidal prior means [batch_size, seq_len, n_features]
            var_p: Sinusoidal prior variances [batch_size, seq_len, n_features]
            yield_pred: Predicted yield values [batch_size, 1]
            target_yield: Ground truth yield values [batch_size, 1]
            weather_feature_mask: Boolean mask indicating predicted positions

        Returns:
            Dictionary containing all loss components
        """
        # Average over batch and multiply by β
        beta = self._current_beta()
        # 1. Reconstruction term: mean over batch of (y_j - μ_θ(z_j))²
        yield_loss = self.criterion(yield_pred.squeeze(), target_yield.squeeze())

        reconstruction_loss = beta * torch.mean(
            torch.sum(
                (weather * (~weather_feature_mask) - z * (~weather_feature_mask)) ** 2,
                dim=(1, 2),
            )
        )
        # 2. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is mixture prior
        kl_term, log_variance = self._compute_sinusoidal_kl_divergence(
            mu_x, var_x, mu_p, var_p, weather_feature_mask
        )
        kl_term = beta * kl_term

        # Total loss: sum of both terms
        total_loss = yield_loss + kl_term

        if log_losses or DRY_RUN:
            print(f"Yield loss: {yield_loss.item()}")
            print(f"Reconstruction loss: {reconstruction_loss.item()}")
            print(f"KL term: {kl_term.item()}")
            print(f"Log variance: {log_variance.item()}")
            print(f"Total loss: {total_loss.item()}")

        return {
            "total_loss": total_loss,
            "yield": yield_loss,
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
        """Compute sinusoidal variational training loss for WeatherFormerSinusoid yield prediction."""

        # Forward pass through WeatherFormerSinusoid model
        # Returns (yield_pred, z, mu_x, var_x, mu_p, var_p)
        yield_pred, z, mu_x, var_x, mu_p, var_p = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
        )

        # Compute all loss components using the helper method
        return self._compute_sinusoidal_variational_loss_components(
            padded_weather,
            z,
            mu_x,
            var_x,
            mu_p,
            var_p,
            yield_pred,
            target_yield,
            weather_feature_mask,
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
        """Compute sinusoidal variational validation loss for WeatherFormerSinusoid yield prediction."""

        # Forward pass through WeatherFormerSinusoid model (no gradient computation)
        with torch.no_grad():
            yield_pred, z, mu_x, var_x, mu_p, var_p = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
            )

        # Compute all loss components using the helper method
        components = self._compute_sinusoidal_variational_loss_components(
            padded_weather,
            z,
            mu_x,
            var_x,
            mu_p,
            var_p,
            yield_pred,
            target_yield,
            weather_feature_mask,
        )
        # only return the reconstruction (RMSE) loss for validation
        return {"total_loss": components["reconstruction"] ** 0.5}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherformer_sinusoid_yield_training_loop(args_dict):
    """
    WeatherFormerSinusoid training loop using the WeatherFormerSinusoidYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    # WeatherFormerSinusoid-specific trainer kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    # WeatherFormerSinusoid-specific model kwargs - no additional parameters needed
    extra_model_kwargs = {}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherFormerSinusoidYieldModel,
        trainer_class=WeatherFormerSinusoidYieldTrainer,
        model_name="weatherformer_sinusoid_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
        extra_model_kwargs=extra_model_kwargs,
    )
