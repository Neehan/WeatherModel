import torch
import torch.nn as nn
from typing import Optional

from src.base_models.weather_cnn import WeatherCNN
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS
from src.base_models.base_model import BaseModel


class WeatherCNNYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        mlp_input_dim: int,
        device: torch.device,
        max_len: int,
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=60,  # Set to 60 as specified in khaki et al 2020
        **model_size_params,
    ):
        super().__init__(name)
        self.weather_model = WeatherCNN(
            weather_dim=weather_dim,
            output_dim=output_dim,
            max_len=max_len,
            device=device,
        )

        # ReLU followed by final yield layer
        self.relu = nn.ReLU()
        self.final_yield_layer = nn.Linear(output_dim, 1)

    def load_pretrained(self, pretrained_model: WeatherCNN):
        """
        override the load_pretrained method from BaseModel to load the weather model
        """
        self.weather_model.load_pretrained(pretrained_model)

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
    ):
        weather = self.weather_model(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )
        # Apply ReLU then final yield layer
        weather = self.relu(weather)
        output = self.final_yield_layer(weather)
        return output
