import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Dict, Tuple
from src.utils.constants import DRY_RUN
from src.utils.losses import compute_mixture_kl_divergence
from src.crop_yield.trainers.weatherformer_yield_trainer import (
    WeatherFormerYieldTrainer,
)
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherformer_mixture_yield_model import (
    WeatherFormerMixtureYieldModel,
)


class WeatherFormerMixtureYieldTrainer(WeatherFormerYieldTrainer):
    """
    Trainer class for WeatherFormerMixture-based crop yield prediction models.

    Inherits from WeatherFormerYieldTrainer but overrides KL divergence computation
    to use mixture prior instead of standard normal prior.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_kl_loss(
        self,
        weather_feature_mask: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_k: torch.Tensor,
        var_k: torch.Tensor,
        log_w_k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss using mixture prior with learnable weights."""
        kl_term = compute_mixture_kl_divergence(
            z=z,
            feature_mask=weather_feature_mask,
            mu_x=mu_x,
            var_x=var_x,
            mu_k=mu_k,
            var_k=var_k,
            log_w_k=log_w_k,
        )
        return kl_term


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherformer_mixture_yield_training_loop(args_dict, use_cropnet: bool):
    """
    WeatherFormerMixture training loop using the WeatherFormerMixtureYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    # WeatherFormerMixture-specific trainer and model kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}
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
