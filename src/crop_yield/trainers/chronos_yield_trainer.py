from src.crop_yield.models.chronos_yield_model import ChronosYieldModel
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)


class ChronosYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for crop yield prediction models using Chronos.
    Inherits from WeatherBERTYieldTrainer - no additional changes needed.
    """
    pass


def chronos_yield_training_loop(args_dict, use_cropnet: bool):
    """
    Chronos training loop using the ChronosYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=ChronosYieldModel,
        trainer_class=ChronosYieldTrainer,
        model_name=f"chronos_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
    )