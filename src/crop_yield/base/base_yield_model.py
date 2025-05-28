import torch
import torch.nn as nn
from typing import Optional

from src.models.weatherbert import WeatherBERT
from src.utils.constants import DEVICE, MAX_CONTEXT_LENGTH
from src.models.base_model import BaseModel


class BaseYieldPredictor(BaseModel):
    def __init__(
        self,
        name: str,
        weather_model: BaseModel,
        mlp_input_dim: int,
    ):
        super().__init__(name)
        self.weather_model = weather_model
        self.mlp_input_dim = mlp_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 120),
            nn.GELU(),
            nn.Linear(120, 1),
        )

    def load_pretrained(self, pretrained_model: WeatherBERT):
        """
        override the load_pretrained method from BaseModel to load the weather model
        """
        self.weather_model.load_pretrained(pretrained_model)

    def prepare_weather_input(self, input_data):
        weather, practices, soil, year, coord, y_past = input_data
        batch_size, n_years, n_features, seq_len = weather.size()

        if n_years * seq_len > MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"n_years * seq_len = {n_years * seq_len} is greater than MAX_CONTEXT_LENGTH = {MAX_CONTEXT_LENGTH}"
            )

        weather = weather.transpose(2, 3)  # transpose n_features and seq_len
        weather = weather.view(batch_size, n_years * seq_len, n_features)

        coord = coord[:, 0, :]

        # Expand year to match the sequence length
        # year is [batch_size, n_years], need to repeat each year for seq_len timesteps
        year = year.unsqueeze(2).expand(
            batch_size, n_years, seq_len
        )  # [batch_size, n_years, seq_len]
        year = year.contiguous().view(
            batch_size, n_years * seq_len
        )  # [batch_size, n_years * seq_len]

        # [7, 8, 11, 1, 2, 29] are the closest weather feature ids according to pretraining
        weather_indices = torch.tensor([7, 8, 11, 1, 2, 29])
        padded_weather = torch.zeros(
            (
                batch_size,
                seq_len * n_years,
                n_features,
            ),
            device=DEVICE,
        )
        padded_weather[:, :, weather_indices] = weather
        # create feature mask
        weather_feature_mask = torch.ones(
            n_features,
            dtype=torch.bool,
            device=DEVICE,
        )
        weather_feature_mask[weather_indices] = False

        # create temporal index (weekly data)
        interval = torch.full((batch_size, 1), 7, device=DEVICE)
        return padded_weather, coord, year, interval, weather_feature_mask

    def forward(self, input_data):
        weather, coord, year, interval, weather_feature_mask = (
            self.prepare_weather_input(input_data)
        )

        weather = self.weather_model(
            (weather, coord, year, interval),
            weather_feature_mask=weather_feature_mask,
        )

        weather = weather.view(weather.size(0), -1)
        output = self.mlp(weather)
        return output
