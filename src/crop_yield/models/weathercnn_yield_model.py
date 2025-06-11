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
        n_past_years: int,
        output_dim: int = 60,  # Set to 60 as specified in khaki et al 2020
        **model_size_params,
    ):
        super().__init__(name)
        self.max_len = (n_past_years + 1) * 52
        self.cnn = WeatherCNN(
            weather_dim=weather_dim,
            output_dim=output_dim,
            max_len=self.max_len,
            device=device,
        )

        # MLP for yield prediction (copied from WeatherBERTYieldModel)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_dim + n_past_years + 1, 1),
        )

    def load_pretrained(self, pretrained_model: WeatherCNN):
        """
        override the load_pretrained method from BaseModel to load the weather model
        """
        self.cnn.load_pretrained(pretrained_model)

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        y_past,
    ):
        # dont pass the mask
        weather = self.cnn(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=None,
        )
        mlp_input = torch.cat([weather, y_past], dim=1)
        # Apply MLP
        output = self.mlp(mlp_input)
        return output
