import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Dict, Tuple
from src.utils.constants import DRY_RUN
from src.utils.losses import compute_mixture_kl_divergence
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherformer_mixture_yield_model import (
    WeatherFormerMixtureYieldModel,
)


class WeatherFormerMixtureYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherFormerMixture-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a mixture variational loss function
    that accounts for the mixture prior nature of WeatherFormerMixture's outputs.

    The loss function includes:
    1. Reconstruction term: (y - μ_θ(z))²
    2. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is the Gaussian mixture prior
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

    def kl_div_loss(self, z, mu_x, var_x, mu_k, var_k, weather_feature_mask):
        """Compute KL divergence loss - can be overridden by child classes."""
        kl_term, log_variance = compute_mixture_kl_divergence(
            z, mu_x, var_x, mu_k, var_k, weather_feature_mask, return_log_variance=True
        )
        return kl_term, log_variance

    def _compute_variational_loss_components(
        self,
        weather: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_k: torch.Tensor,  # [k, seq_len, n_features] - mixture means
        var_k: torch.Tensor,  # [k, seq_len, n_features] - mixture variances
        yield_pred: torch.Tensor,  # [batch_size, 1]
        target_yield: torch.Tensor,  # [batch_size, 1]
        weather_feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - mask of predicted positions
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the mixture variational loss components for WeatherFormerMixture yield prediction.

        Implements the mixture ELBO loss:
        L_yield = ||y_j - μ_θ(z_j)||² + β · KL(q(z_j|x_j) || p(z_j))

        where p(z_j) is the Gaussian mixture prior and KL divergence is computed only on masked features.

        Args:
            z: Sampled latent representations [batch_size, seq_len, n_features]
            mu_x: Posterior means [batch_size, seq_len, n_features]
            var_x: Posterior variances [batch_size, seq_len, n_features]
            mu_k: Mixture component means [k, seq_len, n_features]
            var_k: Mixture component variances [k, seq_len, n_features]
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

        reconstruction_loss = (
            beta
            * torch.sum(
                (weather - z) ** 2 * (~weather_feature_mask),  # keep only input
                dim=(1, 2),
            ).mean()  # mean over batch
        )
        # 2. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is mixture prior
        kl_term, log_variance = self.kl_div_loss(
            z, mu_x, var_x, mu_k, var_k, weather_feature_mask
        )
        kl_term = beta * kl_term

        # Total loss: sum of both terms
        total_loss = yield_loss + reconstruction_loss + kl_term

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
        """Compute mixture variational training loss for WeatherFormerMixture yield prediction."""

        # Forward pass through WeatherFormerMixture model
        # Returns (yield_pred, z, mu_x, var_x, mu_k, var_k)
        yield_pred, z, mu_x, var_x, mu_k, var_k = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
        )

        # Compute all loss components using the helper method
        return self._compute_variational_loss_components(
            padded_weather,
            z,
            mu_x,
            var_x,
            mu_k,
            var_k,
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
        """Compute mixture variational validation loss for WeatherFormerMixture yield prediction."""

        # Forward pass through WeatherFormerMixture model (no gradient computation)
        with torch.no_grad():
            yield_pred, z, mu_x, var_x, mu_k, var_k = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
            )

        # Compute all loss components using the helper method
        components = self._compute_variational_loss_components(
            padded_weather,
            z,
            mu_x,
            var_x,
            mu_k,
            var_k,
            yield_pred,
            target_yield,
            weather_feature_mask,
        )
        # only return the yield (RMSE) loss for validation
        return {"total_loss": components["yield"] ** 0.5}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherformer_mixture_yield_training_loop(args_dict):
    """
    WeatherFormerMixture training loop using the WeatherFormerMixtureYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    # WeatherFormerMixture-specific trainer kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    # WeatherFormerMixture-specific model kwargs
    extra_model_kwargs = {"k": args_dict["n_mixture_components"]}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherFormerMixtureYieldModel,
        trainer_class=WeatherFormerMixtureYieldTrainer,
        model_name="weatherformer_mixture_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
        extra_model_kwargs=extra_model_kwargs,
    )
