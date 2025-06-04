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
import math


class WeatherFormerYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherFormer-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a variational loss function
    that accounts for the probabilistic nature of WeatherFormer's (mu, sigma) outputs.

    The loss function includes:
    1. Reconstruction term: MSE(y_pred, target)
    2. KL divergence term: β * KL(N(mu_x, var_x) || (1/K) * Σ_k N(mu_k, var_k))

    Where:
    - N(mu_x, var_x) is the diagonal normal distribution from the encoder
    - (1/K) * Σ_k N(mu_k, var_k) is an equal-weight mixture of K diagonal Gaussians
    - β is a hyperparameter controlling the strength of the KL regularization
    """

    def __init__(self, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction="mean")
        self.beta = beta
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
            self.output_json["model_config"]["beta"] = beta

    def get_model_name(self) -> str:
        return "weatherformer_yield"

    def _compute_variational_loss_components(
        self,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_k: torch.Tensor,
        var_k: torch.Tensor,
        yield_pred: torch.Tensor,
        target_yield: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the variational loss components for WeatherFormer yield prediction.

        Implements the exact formula from the paper:
        L_yield = MSE(y_pred, target) + β * KL(N(mu_x, sigma_x) || (1/K) * Σ_k N(mu_k, sigma_k))

        Args:
            z: Latent variable samples from q(z) [batch_size, seq_len, output_dim]
            mu_x: Mean of z [batch_size, seq_len, output_dim]
            var_x: diagonal variance of z [batch_size, seq_len, output_dim]
            mu_k: Mean of the K gaussian mixture components [K, seq_len, output_dim]
            var_k: diagonal variance of the K gaussian mixture components [K, seq_len, output_dim]
            yield_pred: Predicted yield values [batch_size, 1]
            target_yield: Ground truth yield values [batch_size, 1]

        Returns:
            Dictionary containing all loss components
        """
        # 1. Reconstruction term: MSE between predicted and target yield
        reconstruction_loss = self.criterion(
            yield_pred.squeeze(), target_yield.squeeze()
        )

        # 2. KL divergence term: KL(q(z) || p(z)) where:
        #    q(z) = N(mu_x, diag(var_x))
        #    p(z) = (1/K) * Σ_k N(mu_k, diag(var_k))

        # Compute log q(z) = log N(z; mu_x, diag(var_x))
        # Sum over seq_len and weather_dim dimensions
        log_q_z = -0.5 * torch.sum(
            torch.log(2 * torch.pi * var_x) + (z - mu_x) ** 2 / var_x, dim=(1, 2)
        )  # dim: [batch size]

        # Compute log p(z) = log((1/K) * Σ_k N(z; mu_k, diag(var_k)))
        # For each mixture component k, compute log N(z; mu_k, diag(var_k))
        n_mixture_components = mu_k.size(0)

        # Reshape for broadcasting: z [batch_size, seq_len, weather_dim] with mu_k [K, seq_len, weather_dim]
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, seq_len, weather_dim]
        mu_k_expanded = mu_k.unsqueeze(0)  # [1, K, seq_len, weather_dim]
        var_k_expanded = var_k.unsqueeze(0)  # [1, K, seq_len, weather_dim]

        # Compute log N(z; mu_k, var_k) for all components simultaneously
        log_components = -0.5 * torch.sum(
            torch.log(2 * torch.pi * var_k_expanded)
            + (z_expanded - mu_k_expanded) ** 2 / var_k_expanded,
            dim=(2, 3),  # sum over seq_len and weather_dim
        )  # dim: [batch_size, n_mixture_components]

        # Use logsumexp for numerical stability: log(1/K * sum(exp(log_components)))
        # Equivalent to: logsumexp(log_components) - log(K)
        log_p_z = torch.logsumexp(log_components, dim=1) - math.log(
            n_mixture_components
        )  # [batch_size]

        # KL divergence: E_q[log q(z) - log p(z)]
        kl_divergence = torch.mean(log_q_z - log_p_z)
        kl_term = self.beta * kl_divergence

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
        """
        Compute variational training loss for WeatherFormer yield prediction.

        This method computes the full variational loss including both the reconstruction
        term (MSE) and the KL divergence regularization term.

        Returns:
            Dict containing:
            - 'total_loss': MSE + β * KL divergence
            - 'reconstruction': MSE between predicted and target yield
            - 'kl_term': β * KL(N(mu_x, var_x) || mixture of gaussians)
        """

        # Prepare input data for the model
        input_data = (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
        )

        # Forward pass through WeatherFormer model
        yield_pred, z, mu_x, var_x, mu_k, var_k = self.model(input_data)

        # Compute all loss components using the helper method
        return self._compute_variational_loss_components(
            z, mu_x, var_x, mu_k, var_k, yield_pred, target_yield
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
        """
        Compute validation loss for WeatherFormer yield prediction.

        For validation, only the reconstruction (MSE) loss is returned to avoid
        overfitting to the KL regularization term during model selection.

        Returns:
            Dict containing:
            - 'total_loss': MSE between predicted and target yield (reconstruction only)
        """

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
            yield_pred, z, mu_x, var_x, mu_k, var_k = self.model(input_data)

        # Compute all loss components using the helper method
        components = self._compute_variational_loss_components(
            z, mu_x, var_x, mu_k, var_k, yield_pred, target_yield
        )
        # only return the reconstruction (MSE) loss for validation
        return {"total_loss": components["reconstruction"]}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherformer_yield_training_loop(args_dict):
    """
    WeatherFormer training loop using the WeatherFormerYieldTrainer class.

    This function sets up and runs cross-validation training for WeatherFormer-based
    yield prediction models. It initializes the model internally and handles all training.

    Args:
        args_dict: Dictionary containing training configuration. Must include:
            - 'beta': Float hyperparameter for KL divergence regularization strength
            - All other parameters required by the base yield training setup

    Returns:
        Training results from cross-validation including model performance metrics
        and trained models.
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
