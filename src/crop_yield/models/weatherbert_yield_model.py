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
        # self.yield_model = WeatherCNNYieldModel(
        #     name=f"{name}_cnn",
        #     device=device,
        #     weather_dim=weather_dim,
        #     n_past_years=n_past_years,
        #     **model_size_params,
        # )
        expected_len = 52 * (n_past_years + 1)
        self.weather_fc = nn.Sequential(
            nn.Linear(
                weather_dim * expected_len, 60
            ),  # flatten weather and make it same dim as cnn output
            nn.GELU(),
        )
        self.yield_mlp = nn.Sequential(
            nn.Linear(60 + n_past_years + 1, 120),  # n_past_years + 1 past yields
            nn.GELU(),
            nn.Linear(120, 1),
        )

    def yield_model(self, weather, coord, year, interval, weather_feature_mask, y_past):
        weather = weather.view(weather.size(0), -1)  # flatten weather
        weather_fc_out = self.weather_fc(weather)
        mlp_input = torch.cat([weather_fc_out, y_past], dim=1)
        return self.yield_mlp(mlp_input)

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

    def forward(self, weather, coord, year, interval, weather_feature_mask, y_past):
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
        # we imputed weather, the mask is not necessary
        output = self.yield_model(
            weather, coord, year, interval, weather_feature_mask=None, y_past=y_past
        )
        return output
