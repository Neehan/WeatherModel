import torch
from typing import Dict, Tuple
from src.crop_yield.trainers.weatherformer_mixture_yield_trainer import (
    WeatherFormerMixtureYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherautoencoder_mixture_yield_model import (
    WeatherAutoencoderMixtureYieldModel,
)


class WeatherAutoencoderMixtureYieldTrainer(WeatherFormerMixtureYieldTrainer):
    """
    Trainer class for WeatherAutoencoderMixture-based crop yield prediction models.

    Inherits from WeatherFormerMixtureYieldTrainer to use the same mixture variational loss function
    but uses WeatherAutoencoderMixtureYieldModel instead of WeatherFormerMixtureYieldModel.
    """


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================
def weatherautoencoder_mixture_yield_training_loop(args_dict):
    """
    WeatherAutoencoderMixture training loop using the WeatherAutoencoderMixtureYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    # WeatherAutoencoderMixture-specific trainer kwargs
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    # WeatherAutoencoderMixture-specific model kwargs
    extra_model_kwargs = {"k": args_dict["n_mixture_components"]}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherAutoencoderMixtureYieldModel,
        trainer_class=WeatherAutoencoderMixtureYieldTrainer,
        model_name="weatherautoencoder_mixture_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
        extra_model_kwargs=extra_model_kwargs,
    )
