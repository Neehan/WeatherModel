import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import pandas as pd
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
from src.crop_yield.models.weathercnn_yield_model import WeatherCNNYieldModel
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)
from src.utils.constants import TOTAL_WEATHER_VARS


class WeatherCNNYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for crop yield prediction models using WeatherCNN.
    Inherits from WeatherBERTYieldTrainer since the training logic is identical.
    """

    pass  # All functionality inherited from WeatherBERTYieldTrainer


def weathercnn_yield_training_loop(args_dict):
    """
    CNN training loop using the WeatherCNNYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=WeatherCNNYieldModel,
        trainer_class=WeatherCNNYieldTrainer,
        model_name="weathercnn_yield",
        args_dict=args_dict,
        extra_model_kwargs={
            "max_len": 52 * (args_dict["n_past_years"] + 1),
        },  # Match the model's output_dim
    )
