from typing import Optional, Union

import torch
import torch.nn as nn
import os

from src.base_models.base_model import BaseModel
from src.crop_yield.models.weathercnn_yield_model import WeatherCNNYieldModel
from src.pretraining.models.weatherbert import WeatherBERT
from src.yield_pretraining.models.seq_model import SeqModel
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS


class WeatherBERTYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        crop_type: str,
        **model_size_params,
    ):
        super().__init__(name)
        self.crop_type = crop_type
        self.device = device
        self.weather_model = WeatherBERT(
            weather_dim=weather_dim,
            output_dim=weather_dim,
            device=device,
            **model_size_params,
        )

        # Initialize seq model for y_past processing
        self.seq_model = SeqModel(name=f"seq_{crop_type}_yield_model", device=device)

        # Attention mechanism to reduce sequence dimension
        self.weather_attention = nn.Sequential(
            nn.Linear(weather_dim, 16), nn.GELU(), nn.Linear(16, 1)
        )

        self.yield_mlp = nn.Sequential(
            nn.Linear(weather_dim + n_past_years + 1, 120),  # weather_dim + past yields
            nn.GELU(),
            nn.Linear(120, 1),
        )

        self.weather_model_frozen = False

    def _load_pretrained_seq_model(self):
        crop_type = self.crop_type.lower()
        if crop_type == "soybeans":
            crop_type = "soybean"
        if crop_type == "winterwheat":
            crop_type = "winter_wheat"
        else:
            crop_type = None
        if crop_type is not None:
            seq_model_path = f"data/trained_models/yield_pretraining/seq_{crop_type}_yield_model_38.4k_latest.pth"
            self.logger.info(f"Loading pretrained seq model from {seq_model_path}")
            checkpoint = torch.load(seq_model_path, map_location=self.device)
            self.seq_model.load_pretrained(checkpoint)

    def _get_seq_output(self, year, coord, period, y_past):
        batch_size, n_past_years = y_past.shape
        year = year.reshape(batch_size, n_past_years, -1)
        coord = coord.unsqueeze(1).expand(batch_size, n_past_years - 1, 2)

        # now keep only one entry per year
        year = year[:, :-1, 0]
        y_past = y_past[:, :-1]
        period = torch.ones(
            batch_size, n_past_years - 1, device=y_past.device
        )  # batch_size x n_past_years (yearly intervals)

        # Get seq model prediction
        seq_output = self.seq_model(year, coord, period, y_past)  # batch_size x 1
        # batch size x num years
        pred_y_past = torch.cat([y_past, seq_output], dim=1)
        return pred_y_past

    def yield_model(self, weather, coord, year, interval, weather_feature_mask, y_past):
        """
        weather: batch_size x seq_len x weather_dim
        coord: batch_size x 2
        year: batch_size x seq_len
        interval: batch_size x 1
        weather_feature_mask: batch_size x seq_len x weather_dim
        y_past: batch_size x n_past_years
        """

        # first predict current year's yield from past yield alone
        y_past_augmented = self._get_seq_output(year, coord, interval, y_past)
        # Apply attention to reduce sequence dimension
        # Compute attention weights
        attention_weights = self.weather_attention(weather)  # batch_size x seq_len x 1
        attention_weights = torch.softmax(
            attention_weights, dim=1
        )  # normalize across sequence

        # Apply attention to get weighted sum
        weather_attended = torch.sum(
            weather * attention_weights, dim=1
        )  # batch_size x weather_dim
        mlp_input = torch.cat([weather_attended, y_past_augmented], dim=1)
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

    def load_pretrained(
        self, pretrained_model: Union[WeatherBERT, "WeatherBERTYieldModel"]
    ):
        """
        override the load_pretrained method from BaseModel to load the weather model
        """
        self.logger.info(f"provided model class: {pretrained_model.__class__.__name__}")
        if isinstance(pretrained_model, WeatherBERT):
            weather_model = pretrained_model
        elif isinstance(pretrained_model, WeatherBERTYieldModel):
            weather_model = pretrained_model.weather_model
        else:
            raise ValueError(
                f"provided model class: {pretrained_model.__class__.__name__} is not supported"
            )

        self.weather_model.load_pretrained(weather_model, load_out_proj=True)
        self._load_pretrained_seq_model()

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
        predicted_weather = self._impute_weather(
            weather, predicted_weather, weather_feature_mask
        )

        output = self.yield_model(
            predicted_weather,
            coord,
            year,
            interval,
            weather_feature_mask=None,
            y_past=y_past,
        )
        return output

    def freeze_weather_model(self):
        if not self.weather_model_frozen:
            self.logger.info("Freezing weather model")
            for param in self.weather_model.parameters():
                param.requires_grad = False
            self.weather_model_frozen = True

    def unfreeze_weather_model(self):
        if self.weather_model_frozen:
            self.logger.info("Unfreezing weather model")
            for param in self.weather_model.parameters():
                param.requires_grad = True
            self.weather_model_frozen = False
