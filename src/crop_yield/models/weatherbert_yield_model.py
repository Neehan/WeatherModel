import torch
import torch.nn as nn
from typing import Optional

from src.pretraining.models.weatherbert import WeatherBERT
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS
from src.base_models.base_model import BaseModel


class WeatherBERTYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        mlp_input_dim: int,
        device: torch.device,
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        **model_size_params,
    ):
        super().__init__(name)
        self.weather_model = WeatherBERT(
            weather_dim=weather_dim,
            output_dim=output_dim,
            device=device,
            **model_size_params,
        )
        self.mlp_input_dim = mlp_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
        )

    def load_pretrained(self, pretrained_model: WeatherBERT):
        """
        override the load_pretrained method from BaseModel to load the weather model
        """
        self.weather_model.load_pretrained(pretrained_model)

    def _impute_weather(self, original_weather, imputed_weather, weather_feature_mask):
        """
        Fast combination using element-wise ops instead of torch.where:
        - original_weather: batch_size x seq_len x weather_dim
        - imputed_weather: batch_size x seq_len x weather_dim
        - weather_feature_mask: batch_size x seq_len x weather_dim
        """
        # return (
        #     original_weather
        #     * (~weather_feature_mask)  # keep original where mask is False
        #     + imputed_weather * weather_feature_mask
        # )
        return imputed_weather

    def forward(self, input_data):
        # (padded_weather, coord_processed, year_expanded, interval, weather_feature_mask, practices, soil, y_past, y)
        padded_weather, coord, year, interval, weather_feature_mask = input_data

        imputed_weather = self.weather_model(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )
        weather = self._impute_weather(
            padded_weather, imputed_weather, weather_feature_mask
        )
        weather = weather.reshape(weather.size(0), -1)
        output = self.mlp(weather)
        return output
