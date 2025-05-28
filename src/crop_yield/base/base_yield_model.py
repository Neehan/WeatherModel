import torch
import torch.nn as nn
from typing import Optional

from src.models.weatherbert import WeatherBERT
from src.utils.constants import DEVICE, MAX_CONTEXT_LENGTH


class BaseYieldPredictor(nn.Module):
    def __init__(
        self,
        pretrained_weather_model: Optional[WeatherBERT],
        weather_model_size_params: dict,
    ):
        super().__init__()
        self.weather_model = WeatherBERT(
            31, 31, max_len=MAX_CONTEXT_LENGTH, **weather_model_size_params
        )
        if pretrained_weather_model is not None:
            self.weather_model.load_pretrained(pretrained_weather_model)

        self.mlp = nn.Sequential(
            nn.Linear(31 * MAX_CONTEXT_LENGTH, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
        )

    def forward(self, input_data):
        weather, practices, soil, year, coord, y_past, mask = input_data

        batch_size, n_years, n_features, seq_len = weather.size()
        weather = weather.view(batch_size * n_years, -1, n_features)

        coord = coord.view(batch_size * n_years, 2)
        year = year.view(batch_size * n_years, 1)
        # [7, 8, 11, 1, 2, 29] are the closest weather feature ids according to pretraining
        weather_indices = torch.tensor([7, 8, 11, 1, 2, 29])
        padded_weather = torch.zeros(
            (
                batch_size * n_years,
                seq_len,
                self.weather_model.input_dim,
            ),
            device=DEVICE,
        )
        padded_weather[:, :, weather_indices] = weather
        # create feature mask
        weather_feature_mask = torch.ones(
            self.weather_model.input_dim,
            dtype=torch.bool,
            device=DEVICE,
        )
        weather_feature_mask[weather_indices] = False

        # create temporal index (weekly data)
        temporal_gran = torch.full((batch_size * n_years, 1), 7, device=DEVICE)
        temporal_index = torch.cat([year, temporal_gran], dim=1)

        weather = self.weather_model(
            (padded_weather, coord, temporal_index),
            weather_feature_mask=weather_feature_mask,
        )

        weather = weather.view(batch_size * n_years, -1)
        output = self.mlp(weather)
        return output
