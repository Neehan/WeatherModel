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
    Inherits from WeatherBERTYieldTrainer but overrides loss computation to pass soil data.
    """

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
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for CNN-RNN model - passes soil data to the model."""
        # Forward pass through the CNN-RNN model with soil data
        predicted_yield = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            soil,
            y_past,
        )

        # Compute MSE loss
        loss = self.criterion(predicted_yield.squeeze(), target_yield.squeeze())

        return {"total_loss": loss}

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
    ) -> Dict[str, torch.Tensor]:
        """Compute validation loss for CNN-RNN model - passes soil data to the model."""

        # Forward pass through the CNN-RNN model (no gradient computation needed for validation)
        with torch.no_grad():
            predicted_yield = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                soil,
                y_past,
            )

        # Return RMSE for validation since that's standard for comparison
        loss = self.criterion(predicted_yield.squeeze(), target_yield.squeeze())
        return {"total_loss": loss**0.5}


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
