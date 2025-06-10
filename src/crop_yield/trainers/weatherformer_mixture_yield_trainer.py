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
                    "reconstruction": [],
                    "log_variance": [],
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],  # just MSE
                },
            }

    def _compute_mixture_kl_divergence(
        self,
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_k: torch.Tensor,  # [k, seq_len, n_features] - mixture means
        var_k: torch.Tensor,  # [k, seq_len, n_features] - mixture variances
        weather_feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - weather feature mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence between posterior q(z|x) and mixture prior p(z).

        KL(q(z|x) || p(z)) = log q(z|x) - log p(z)

        where:
        - log q(z|x) = -D/2 log(2π) - 1/2 Σ_d [log σ²_φ,d(x) + (z_d - μ_φ,d(x))²/σ²_φ,d(x)]
        - log p(z) = -log K - D/2 log(2π) + log(Σ_k exp(-1/2 Σ_d [log σ²_k,d + (z_d - μ_k,d)²/σ²_k,d]))
        """
        # Compute log q(z|x) - posterior log-density
        # log q(z|x) = -D/2 log(2π) - 1/2 Σ_d [log σ²_φ,d + (z_d - μ_φ,d)²/σ²_φ,d]
        D = weather_feature_mask.sum(dim=(1, 2))  # number of active dimensions
        log_variance = torch.log(var_x)
        log_q_z_x = -0.5 * (log_variance + (z - mu_x) ** 2 / var_x)
        # apply mask (keep only for masked out indices)
        log_q_z_x = log_q_z_x * weather_feature_mask
        log_q_z_x = torch.sum(
            log_q_z_x,
            dim=(1, 2),  # sum over seq_len and n_features
        )  # [batch_size]
        log_q_z_x = log_q_z_x - 0.5 * D * math.log(2 * math.pi)

        # Compute log p(z) - mixture prior log-density
        # First compute log-density for each mixture component
        z_expanded = z.unsqueeze(0)  # [1, batch_size, seq_len, n_features]
        mu_k_expanded = mu_k.unsqueeze(1)  # [k, 1, seq_len, n_features]
        var_k_expanded = var_k.unsqueeze(1)  # [k, 1, seq_len, n_features]
        mask_expanded = weather_feature_mask.unsqueeze(
            0
        )  # [1, batch_size, seq_len, n_features]

        # Compute log-density for each component: [k, batch_size]
        log_component_densities = -0.5 * (
            torch.log(var_k_expanded)
            + (z_expanded - mu_k_expanded) ** 2 / var_k_expanded
        )
        # apply mask
        log_component_densities = log_component_densities * mask_expanded
        log_component_densities = torch.sum(
            log_component_densities,
            dim=(2, 3),  # sum over seq_len and n_features
        )  # [k, batch_size]

        # Compute log p(z) = log(1/K * Σ_k exp(log_component_densities))
        # = -log K + log(Σ_k exp(log_component_densities))
        k = mu_k.shape[0]  # number of mixture components
        log_p_z = torch.logsumexp(log_component_densities, dim=0) - math.log(
            k
        )  # [batch_size]

        # KL divergence: KL(q(z|x) || p(z)) = log q(z|x) - log p(z)
        kl_divergence = log_q_z_x - log_p_z  # [batch_size]

        log_variance = self._masked_mean(log_variance, weather_feature_mask, dim=(1, 2))

        return kl_divergence.mean(), log_variance.mean()  # average over batch

    def _compute_mixture_variational_loss_components(
        self,
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_k: torch.Tensor,  # [k, seq_len, n_features] - mixture means
        var_k: torch.Tensor,  # [k, seq_len, n_features] - mixture variances
        yield_pred: torch.Tensor,  # [batch_size, 1]
        target_yield: torch.Tensor,  # [batch_size, 1]
        weather_feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - weather feature mask
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the mixture variational loss components for WeatherFormerMixture yield prediction.

        Implements the mixture ELBO loss:
        L_yield = ||y_j - μ_θ(z_j)||² + β · KL(q(z_j|x_j) || p(z_j))

        where p(z_j) is the Gaussian mixture prior.

        Args:
            z: Sampled latent representations [batch_size, seq_len, n_features]
            mu_x: Posterior means [batch_size, seq_len, n_features]
            var_x: Posterior variances [batch_size, seq_len, n_features]
            mu_k: Mixture component means [k, seq_len, n_features]
            var_k: Mixture component variances [k, seq_len, n_features]
            yield_pred: Predicted yield values [batch_size, 1]
            target_yield: Ground truth yield values [batch_size, 1]

        Returns:
            Dictionary containing all loss components
        """
        # 1. Reconstruction term: mean over batch of (y_j - μ_θ(z_j))²
        reconstruction_loss = self.criterion(
            yield_pred.squeeze(), target_yield.squeeze()
        )

        # 2. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is mixture prior
        kl_term, log_variance = self._compute_mixture_kl_divergence(
            z, mu_x, var_x, mu_k, var_k, weather_feature_mask
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
        """Compute mixture variational training loss for WeatherFormerMixture yield prediction."""

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherFormerMixture model
        # Returns (yield_pred, z, mu_x, var_x, mu_k, var_k)
        yield_pred, z, mu_x, var_x, mu_k, var_k = self.model(input_data)

        # Compute all loss components using the helper method
        return self._compute_mixture_variational_loss_components(
            z, mu_x, var_x, mu_k, var_k, yield_pred, target_yield, weather_feature_mask
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

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherFormerMixture model (no gradient computation)
        with torch.no_grad():
            yield_pred, z, mu_x, var_x, mu_k, var_k = self.model(input_data)

        # Compute all loss components using the helper method
        components = self._compute_mixture_variational_loss_components(
            z, mu_x, var_x, mu_k, var_k, yield_pred, target_yield, weather_feature_mask
        )
        # only return the reconstruction (RMSE) loss for validation
        return {"total_loss": components["reconstruction"] ** 0.5}


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
