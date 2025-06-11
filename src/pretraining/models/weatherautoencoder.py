import torch
import torch.nn as nn
from typing import Optional
from src.pretraining.models.weatherbert import WeatherBERT
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE

"""
This class implements the WeatherAutoencoder model for baseline comparison.

The output is just mu (predicted values), unlike WeatherFormer which outputs (mu, sigma).
Uses standard MSE loss for training.
"""


class WeatherAutoencoder(WeatherBERT):
    def __init__(
        self,
        weather_dim,
        output_dim,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
        device=DEVICE,
    ):
        super(WeatherAutoencoder, self).__init__(
            weather_dim=weather_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim_factor=hidden_dim_factor,
            max_len=max_len,
            device=device,
        )
        # override the name
        self.name = "weatherautoencoder"
