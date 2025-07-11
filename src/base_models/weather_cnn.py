import torch
import torch.nn as nn
import copy
from typing import Optional

from src.base_models.base_model import BaseModel
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE, TOTAL_WEATHER_VARS


class WeatherCNN(BaseModel):
    def __init__(
        self,
        weather_dim,
        output_dim,
        max_len,
        device=DEVICE,
    ):
        super(WeatherCNN, self).__init__("weathercnn")

        self.weather_dim = weather_dim
        self.input_dim = weather_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.device = device

        # Assume weekly data: 52 weeks per year
        self.weeks_per_year = 52
        self.n_years = max_len // self.weeks_per_year

        # CNN layers for processing weather sequences (per year) - matching original paper
        self.weather_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=9, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=16, out_channels=20, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Calculate the final sequence length after all CNN operations
        # Use a dummy input to calculate CNN output features per year
        dummy_input = torch.zeros(1, 1, self.weeks_per_year)
        with torch.no_grad():
            dummy_output = self.weather_cnn(dummy_input)
        self.cnn_features_per_feature = dummy_output.shape[1]
        self.cnn_features_per_year = self.cnn_features_per_feature * self.weather_dim

        # FC layer to combine features per year (matching original paper)
        self.weather_fc = nn.Sequential(
            nn.Linear(self.cnn_features_per_year, output_dim), nn.ReLU()
        )

    def load_pretrained(self, pretrained_model: "WeatherCNN"):
        """Load weights from a pretrained WeatherCNN model by deep copying each layer."""

        if self.weather_dim != pretrained_model.weather_dim:
            raise ValueError(
                f"expected weather dimension {self.weather_dim} but received {pretrained_model.weather_dim}"
            )
        if self.output_dim != pretrained_model.output_dim:
            raise ValueError(
                f"expected output dimension {self.output_dim} but received {pretrained_model.output_dim}"
            )
        if self.max_len != pretrained_model.max_len:
            raise ValueError(
                f"expected max length {self.max_len} but received {pretrained_model.max_len}"
            )
        if self.cnn_features_per_year != pretrained_model.cnn_features_per_year:
            raise ValueError(
                f"expected CNN features per year {self.cnn_features_per_year} but received {pretrained_model.cnn_features_per_year}"
            )

        self.weather_cnn = copy.deepcopy(pretrained_model.weather_cnn)
        self.weather_fc = copy.deepcopy(pretrained_model.weather_fc)

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process weather data through CNN layers.
        weather: batch_size x seq_len x n_features
        Returns: batch_size x n_years x output_dim
        """
        batch_size, seq_len, n_features = weather.shape
        if n_features != self.weather_dim:
            raise ValueError(
                f"expected {self.weather_dim} weather features but received {n_features} features"
            )

        # Apply masking if provided
        if weather_feature_mask is not None:
            weather = weather * (~weather_feature_mask)

        # Reshape to process per year: (batch_size, n_years, weeks_per_year, weather_dim)
        weather = weather.view(
            batch_size, self.n_years, self.weeks_per_year, self.weather_dim
        )

        # Reshape to process each feature of each year separately: (batch_size * n_years * weather_dim, 1, weeks_per_year)
        weather = weather.reshape(
            batch_size * self.n_years * self.weather_dim, 1, self.weeks_per_year
        )

        # Pass each feature of each year through CNN
        cnn_output = self.weather_cnn(
            weather
        )  # (batch_size * n_years * weather_dim, cnn_features_per_feature)

        # Reshape to combine features per year: (batch_size * n_years, cnn_features_per_year)
        cnn_output = cnn_output.view(
            batch_size * self.n_years, self.cnn_features_per_year
        )

        # Apply FC layer to combine features per year
        year_features = self.weather_fc(
            cnn_output
        )  # (batch_size * n_years, output_dim)

        # Reshape back to separate years: (batch_size, n_years, output_dim)
        year_features = year_features.view(batch_size, self.n_years, self.output_dim)

        return year_features
