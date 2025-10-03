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
    Trainer class for USDA-style linear regression crop yield model.
    Simple MSE loss, no regularization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction="mean")

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
    ):
        """
        Compute training loss: simple MSE
        """
        yield_pred = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
            soil,
        )

        mse_loss = self.criterion(yield_pred.squeeze(), target_yield.squeeze())

        return {"total_loss": mse_loss}

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
    ):
        """
        Compute validation loss: RMSE
        """
        yield_pred = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
            soil,
        )

        mse_loss = self.criterion(yield_pred.squeeze(), target_yield.squeeze())

        return {"total_loss": mse_loss**0.5}


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
        extra_model_kwargs={"crop_type": args_dict["crop_type"]},
    )
