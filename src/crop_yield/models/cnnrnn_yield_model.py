import torch
import torch.nn as nn
from typing import Optional

from src.base_models.weather_cnn import WeatherCNN
from src.base_models.soil_cnn import SoilCNN
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS
from src.base_models.base_model import BaseModel
from src.utils.utils import normalize_year_interval_coords


class CNNRNNYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        output_dim: int = 60,  # CNN output features
        soil_output_dim: int = 40,  # Soil CNN output features
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        **model_size_params,
    ):
        super().__init__(name)
        self.max_len = (n_past_years + 1) * 52
        self.n_past_years = n_past_years
        self.device = device

        # CNN for processing weather data
        self.cnn = WeatherCNN(
            weather_dim=weather_dim,
            output_dim=output_dim,
            max_len=self.max_len,
            device=device,
        )

        # CNN for processing soil data
        self.soil_cnn = SoilCNN()

        # LSTM for processing temporal sequence with past yields, coordinates, weather, and soil features
        # Input: CNN output + soil CNN output + past yields + coordinates + year
        lstm_input_dim = (
            output_dim + soil_output_dim + 2 + 1 + 1
        )  # weather CNN output + soil CNN output + coords + year + past_yield
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.0 if lstm_num_layers == 1 else 0.1,
        )

        # Final prediction layer
        self.output_layer = nn.Linear(lstm_hidden_dim, 1)

    def load_pretrained(self, pretrained_model: WeatherCNN):
        """
        Load pretrained CNN weights
        """
        self.cnn.load_pretrained(pretrained_model)

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        soil,
        y_past,
    ):
        # Get weather representations from CNN (batch_size, n_years, output_dim)
        weather_features = self.cnn(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )

        # Process soil data through soil CNN (batch_size, n_years, soil_output_dim)
        soil_features = self.soil_cnn(soil)

        batch_size, n_years, cnn_output_dim = weather_features.shape

        # Normalize year, interval, and coords (following original paper)
        year_norm, interval_norm, coords_norm = normalize_year_interval_coords(
            year, interval, coord
        )

        # Prepare coordinates - expand to match sequence length
        coords_expanded = coords_norm.unsqueeze(1).expand(batch_size, n_years, 2)

        # Prepare year - take mean year for each sequence and expand
        year_mean = year_norm.mean(dim=1, keepdim=True)  # (batch_size, 1)
        year_expanded = year_mean.unsqueeze(2).expand(batch_size, n_years, 1)

        # Prepare past yields - expand to match sequence length
        # y_past should be (batch_size, n_past_years + 1)
        y_past_expanded = y_past.unsqueeze(2).expand(batch_size, n_years, 1)

        # Concatenate weather CNN features, soil CNN features, coordinates, year, and past yields
        lstm_input = torch.cat(
            [
                weather_features,
                soil_features,
                coords_expanded,
                year_expanded,
                y_past_expanded,
            ],
            dim=2,
        )  # (batch_size, n_years, lstm_input_dim)

        # Pass through LSTM
        lstm_output, _ = self.lstm(lstm_input)  # (batch_size, n_years, lstm_hidden_dim)

        # Take the last timestep output for prediction
        final_output = lstm_output[:, -1, :]  # (batch_size, lstm_hidden_dim)

        # Final prediction
        prediction = self.output_layer(final_output)  # (batch_size, 1)

        return prediction
