import torch
import torch.nn as nn
from typing import Optional

from src.pretraining.models.weatherbert import WeatherBERT
from src.crop_yield.models.weathercnn_yield_model import WeatherCNNYieldModel
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS
from src.base_models.base_model import BaseModel


class WeatherBERTYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        super().__init__(name)
        self.weather_model = WeatherBERT(
            weather_dim=weather_dim,
            output_dim=weather_dim,
            device=device,
            **model_size_params,
        )
        self.yield_model = WeatherCNNYieldModel(
            name=f"{name}_yield",
            device=device,
            weather_dim=weather_dim,
            n_past_years=n_past_years,
            **model_size_params,
        )

    def _impute_weather(self, original_weather, imputed_weather, weather_feature_mask):
        """
        Fast combination using element-wise ops instead of torch.where:
        - original_weather: batch_size x seq_len x weather_dim
        - imputed_weather: batch_size x seq_len x weather_dim
        - weather_feature_mask: batch_size x seq_len x weather_dim
        """
        return (
            original_weather
            * (~weather_feature_mask)  # keep original where mask is False
            + imputed_weather * weather_feature_mask
        )

    def load_pretrained(self, pretrained_model: WeatherBERT):
        """
        override the load_pretrained method from BaseModel to load the weather model
        """
        self.weather_model.load_pretrained(pretrained_model)

    def forward(self, weather, coord, year, interval, weather_feature_mask):
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2 (lat, lon) UNNORMALIZED
        year: batch_size x seq_len (UNNORMALIZED, time-varying years)
        interval: batch_size x 1 (UNNORMALIZED in days)
        weather_feature_mask: batch_size x seq_len x n_features
        """

        predicted_weather = self.weather_model(
            weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )

        # Fast combination using element-wise ops
        weather = self._impute_weather(weather, predicted_weather, weather_feature_mask)
        output = self.yield_model(weather, coord, year, interval, weather_feature_mask)
        return output
