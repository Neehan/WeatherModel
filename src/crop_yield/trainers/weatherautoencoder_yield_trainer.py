from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.weatherautoencoder_yield_model import (
    WeatherAutoencoderYieldModel,
)


class WeatherAutoencoderYieldTrainer(WeatherBERTYieldTrainer):
    """
    WeatherAutoencoder yield trainer that inherits from WeatherBERTYieldTrainer.
    Uses the same training logic but with WeatherAutoencoderYieldModel.
    """

    pass  # Inherits all functionality from parent


def weatherautoencoder_yield_training_loop(args_dict, use_cropnet: bool):
    """
    WeatherAutoencoder training loop using the WeatherAutoencoderYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherAutoencoderYieldModel,
        trainer_class=WeatherAutoencoderYieldTrainer,
        model_name="weatherautoencoder_yield",
        args_dict=args_dict,
    )
