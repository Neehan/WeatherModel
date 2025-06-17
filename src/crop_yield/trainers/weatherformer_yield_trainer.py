import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Optional
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherformer_yield_model import WeatherFormerYieldModel
from src.utils.losses import (
    compute_gaussian_kl_divergence,
    gaussian_nll_loss,
    masked_mean,
)


class WeatherFormerYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for WeatherFormer-based crop yield prediction models.

    Inherits from WeatherBERTYieldTrainer but implements a variational loss function
    that accounts for the probabilistic nature of WeatherFormer's (mu, sigma) outputs.

    The loss function includes:
    1. Yield term: MSE between predicted and target yield
    2. Reconstruction term: β * Gaussian NLL for weather features
    3. KL divergence term: β * KL(q(z|x) || p(z)) where p(z) is standard normal
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
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],  # just RMSE
                },
            }

    def compute_kl_loss(
        self,
        weather_feature_mask: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        """Compute KL divergence loss between posterior and standard normal prior.
        Derived classes can override this method for different priors."""

        # Standard normal prior for VAE: mu_p = 0, var_p = 1
        mu_p = torch.zeros_like(mu_x)
        var_p = torch.ones_like(var_x)

        kl_term = compute_gaussian_kl_divergence(
            mu_x=mu_x,
            var_x=var_x,
            mu_p=mu_p,
            var_p=var_p,
            feature_mask=weather_feature_mask,
        )
        return kl_term

    def _compute_variational_loss_components(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        target_yield: torch.Tensor,
        yield_pred: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        *args,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the variational loss components for WeatherFormer yield prediction.

        Args:
            weather: Original weather data [batch_size, seq_len, weather_dim]
            mu_x: Mean of weather representations [batch_size, seq_len, weather_dim]
            var_x: Variance of weather representations [batch_size, seq_len, weather_dim]
            yield_pred: Predicted yield values [batch_size, 1]
            target_yield: Ground truth yield values [batch_size, 1]
            weather_feature_mask: Boolean mask indicating predicted positions

        Returns:
            Dictionary containing all loss components
        """
        # 1. Yield term: MSE between predicted and target yield
        yield_loss = self.criterion(yield_pred.squeeze(), target_yield.squeeze())

        beta = self._current_beta()
        # 2. Reconstruction term: Gaussian NLL for weather features
        # use negative mask so that it is computed for input weather vals
        # reconstruction_loss = beta * gaussian_nll_loss(
        #     weather, mu_x, var_x, ~weather_feature_mask
        # )
        reconstruction_loss = (
            beta
            * masked_mean(
                (weather - z) ** 2,  # keep only input
                ~weather_feature_mask,
                dim=(1, 2),
            ).mean()
        )  # mean over batch

        # 3. KL divergence term: β * KL(q(z|x) || p(z))
        kl_term = beta * self.compute_kl_loss(
            weather_feature_mask, z, mu_x, var_x, *args
        )

        # Total loss: sum of all terms
        total_loss = yield_loss + reconstruction_loss + kl_term

        return {
            "total_loss": total_loss,
            "yield": yield_loss,
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
        # Returns (yield_pred, mu_x, var_x)
        model_outputs = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
        )
        # Compute all loss components using the helper method
        return self._compute_variational_loss_components(
            padded_weather, weather_feature_mask, target_yield, *model_outputs
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
            model_outputs = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
            )

        # Compute all loss components using the helper method
        components = self._compute_variational_loss_components(
            padded_weather, weather_feature_mask, target_yield, *model_outputs
        )
        # only return the yield (RMSE) loss for validation
        return {"total_loss": components["yield"] ** 0.5}


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
