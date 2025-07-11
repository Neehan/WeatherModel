import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import pandas as pd
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
from src.crop_yield.models.cnnrnn_yield_model import CNNRNNYieldModel
from src.crop_yield.trainers.weatherbert_yield_trainer import (
    WeatherBERTYieldTrainer,
    _create_yield_training_setup,
    _run_yield_cross_validation,
)


class CNNRNNYieldTrainer(WeatherBERTYieldTrainer):
    """
    Trainer class for crop yield prediction models using CNN-RNN.
    Inherits from WeatherBERTYieldTrainer since the training logic is identical.
    """

    pass  # All functionality inherited from WeatherBERTYieldTrainer


def cnnrnn_yield_training_loop(args_dict, use_cropnet: bool):
    """
    CNN-RNN training loop using the CNNRNNYieldTrainer class.
    Initializes the model internally and handles all training.
    """
    setup_params = _create_yield_training_setup(args_dict, use_cropnet)

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=CNNRNNYieldModel,
        trainer_class=CNNRNNYieldTrainer,
        model_name=f"cnnrnn_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
    )
