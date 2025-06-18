import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Dict, Tuple
from src.utils.constants import DRY_RUN
from src.utils.losses import compute_gaussian_kl_divergence
from src.crop_yield.trainers.weatherformer_yield_trainer import (
    WeatherFormerYieldTrainer,
)
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherformer_sinusoid_yield_model import (
    WeatherFormerSinusoidYieldModel,
)


class WeatherFormerSinusoidYieldTrainer(WeatherFormerYieldTrainer):
    """
    Trainer class for WeatherFormerSinusoid-based crop yield prediction models.

    Inherits from WeatherFormerYieldTrainer but overrides the KL divergence computation
    to use Gaussian KL divergence with sinusoidal priors instead of standard normal priors.
    """

    def compute_kl_loss(
        self,
        weather_feature_mask: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss using sinusoidal priors instead of standard normal."""
        kl_term = compute_gaussian_kl_divergence(
            feature_mask=weather_feature_mask,
            mu_x=mu_x,
            var_x=var_x,
            mu_p=mu_p,
            var_p=var_p,
        )
        return kl_term


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

    # WeatherFormerSinusoid-specific model kwargs
    extra_model_kwargs = {"k": args_dict["n_mixture_components"]}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherFormerSinusoidYieldModel,
        trainer_class=WeatherFormerSinusoidYieldTrainer,
        model_name="weatherformer_sinusoid_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
        extra_model_kwargs=extra_model_kwargs,
    )
