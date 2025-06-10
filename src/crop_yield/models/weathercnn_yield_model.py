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
        device: torch.device,
        weather_dim: int,
        max_len: int,
        n_past_years: int,
        output_dim: int = 60,  # Set to 60 as specified in khaki et al 2020
        **model_size_params,
    ):
        super().__init__(name)
        self.weather_model = WeatherCNN(
            weather_dim=weather_dim,
            output_dim=output_dim,
            max_len=max_len,
            device=device,
        )

        # MLP for yield prediction (copied from WeatherBERTYieldModel)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim + n_past_years + 1, 120),
            nn.GELU(),
            nn.Linear(120, 1),
        )

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
        practices,
        soil,
        y_past,
    ):
        # (padded_weather, coord_processed, year_expanded, interval, weather_feature_mask)
        weather = self.weather_model(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )
        mlp_input = torch.cat([weather, y_past], dim=1)
        # Apply MLP
        output = self.mlp(mlp_input)
        return output
