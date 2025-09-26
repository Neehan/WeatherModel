import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import pandas as pd
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
from src.crop_yield.models.linear_yield_model import LinearYieldModel
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)


class LinearYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for crop yield prediction models using linear regression.
    Inherits from WeatherBERTYieldTrainer since the training logic is identical.
    """

    pass  # All functionality inherited from WeatherBERTYieldTrainer


def linear_yield_training_loop(args_dict, use_cropnet: bool):
    """
    Linear regression training loop using the LinearYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=LinearYieldModel,
        trainer_class=LinearYieldTrainer,
        model_name=f"linear_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
    )
