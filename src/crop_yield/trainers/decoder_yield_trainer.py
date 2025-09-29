import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Optional

from src.crop_yield.trainers.weatherformer_yield_trainer import (
    WeatherFormerYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.utils.constants import DRY_RUN
from src.crop_yield.models.decoder_yield_model import DecoderYieldModel
from src.utils.losses import (
    compute_gaussian_kl_divergence,
    gaussian_log_likelihood,
)


class DecoderYieldTrainer(WeatherFormerYieldTrainer):
    """
    Trainer class for Decoder-based crop yield prediction models.

    Inherits from WeatherFormerYieldTrainer but implements a modified loss function
    that includes weather reconstruction loss in addition to yield prediction and KL divergence.

    The loss function includes:
    1. Yield term: MSE between predicted and target yield
    2. Weather reconstruction term: β * MSE between predicted and original weather
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
                    "weather_reconstruction": [],
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],  # just RMSE
                },
            }

    def compute_elbo_loss(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        target_yield: torch.Tensor,
        yield_pred: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        weather_pred: torch.Tensor,
        *args,
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the ELBO loss for Decoder yield prediction.

        Args:
            weather: Original weather data
            weather_feature_mask: Boolean mask for weather features
            target_yield: Ground truth yield values
            yield_pred: Predicted yield values
            z: Sampled weather representations
            mu_x: Mean of weather representations
            var_x: Variance of weather representations
            weather_pred: Reconstructed weather from decoder
        """

        # 1. Yield term: MSE between predicted and target yield
        yield_loss = self.criterion(yield_pred.squeeze(), target_yield.squeeze())

        beta = self._current_beta()

        # 2. Weather reconstruction term: β * MSE between predicted and original weather
        # Only compute reconstruction loss for non-masked (input) features
        input_mask = ~weather_feature_mask
        if input_mask.any():
            weather_reconstruction_loss = beta * self.criterion(
                weather_pred[input_mask], weather[input_mask]
            )
        else:
            weather_reconstruction_loss = torch.tensor(0.0, device=weather.device)

        # 3. KL divergence term: β * KL(q(z|x) || p(z))
        kl_term = (
            beta
            * self.compute_kl_loss(weather_feature_mask, z, mu_x, var_x, *args).mean()
        )

        if log_losses or DRY_RUN:
            self.logger.info(f"Yield Loss: {yield_loss.item():.6f}")
            self.logger.info(
                f"Weather Reconstruction Loss: {weather_reconstruction_loss.item():.6f}"
            )
            self.logger.info(f"KL Term: {kl_term.item():.6f}")

        # Total loss: sum of all terms
        total_loss = yield_loss + weather_reconstruction_loss + kl_term

        return {
            "total_loss": total_loss,
            "yield": yield_loss,
            "weather_reconstruction": weather_reconstruction_loss,
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
        """Compute variational training loss for Decoder yield prediction."""

        # Forward pass through Decoder model
        # Returns (yield_pred, z, mu_x, var_x, weather_pred)
        model_outputs = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
        )

        # Compute ELBO loss using the modified helper method
        return self.compute_elbo_loss(
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
        """Compute variational validation loss for Decoder yield prediction."""

        # Forward pass through Decoder model (no gradient computation)
        with torch.no_grad():
            model_outputs = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
            )

        # Compute ELBO loss using the modified helper method
        components = self.compute_elbo_loss(
            padded_weather, weather_feature_mask, target_yield, *model_outputs
        )

        # only return the yield (RMSE) loss for validation
        return {"total_loss": components["yield"] ** 0.5}


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def decoder_yield_training_loop(args_dict, use_cropnet: bool):
    """
    Decoder training loop using the DecoderYieldTrainer class.
    Initializes the model internally and handles all training.

    Args:
        args_dict: Arguments dictionary
        use_cropnet: Whether to use CropNet training
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    # Decoder-specific trainer kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=DecoderYieldModel,
        trainer_class=DecoderYieldTrainer,
        model_name=f"decoder_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
    )
