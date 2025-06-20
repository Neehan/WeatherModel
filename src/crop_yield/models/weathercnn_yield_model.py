import torch
import torch.nn as nn
from typing import Optional

from src.base_models.weather_cnn import WeatherCNN
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

        # Attention mechanism to reduce sequence dimension (identical to WeatherBERT)
        self.weather_attention = nn.Sequential(
            nn.Linear(output_dim, 16), nn.GELU(), nn.Linear(16, 1)
        )

        # MLP for yield prediction (identical to WeatherBERT)
        self.yield_mlp = nn.Sequential(
            nn.Linear(output_dim + n_past_years + 1, 120),  # output_dim + past yields
            nn.GELU(),
            nn.Linear(120, 1),
        )

    def yield_model(self, weather, coord, year, interval, weather_feature_mask, y_past):
        # Apply attention to reduce sequence dimension (identical to WeatherBERT)
        # Compute attention weights
        attention_weights = self.weather_attention(weather)  # batch_size x seq_len x 1
        attention_weights = torch.softmax(
            attention_weights, dim=1
        )  # normalize across sequence

        # Apply attention to get weighted sum
        weather_attended = torch.sum(
            weather * attention_weights, dim=1
        )  # batch_size x output_dim

        mlp_input = torch.cat([weather_attended, y_past], dim=1)
        return self.yield_mlp(mlp_input)

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
        # Get weather representations from CNN (batch_size, n_years, output_dim)
        # it doesnt support mask
        weather = self.cnn(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=None,
        )

        # Use identical yield model as WeatherBERT
        output = self.yield_model(
            weather,
            coord,
            year,
            interval,
            weather_feature_mask=None,
            y_past=y_past,
        )
        return output
