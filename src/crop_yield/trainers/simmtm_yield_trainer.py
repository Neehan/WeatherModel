from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.crop_yield.models.simmtm_yield_model import (
    SimMTMYieldModel,
)


class SimMTMYieldTrainer(WeatherBERTYieldTrainer):
    """
    SimMTM yield trainer that inherits from WeatherBERTYieldTrainer.
    Uses the same training logic but with SimMTMYieldModel.
    """

    pass  # Inherits all functionality from parent


def simmtm_yield_training_loop(args_dict, use_cropnet: bool):
    """
    SimMTM training loop using the SimMTMYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=SimMTMYieldModel,
        trainer_class=SimMTMYieldTrainer,
        model_name=f"simmtm_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
    )
