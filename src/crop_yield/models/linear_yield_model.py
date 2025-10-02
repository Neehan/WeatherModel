import torch
import torch.nn as nn
from typing import Optional

from src.base_models.base_model import BaseModel


class LinearYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        super().__init__(name)
        self.weather_dim = weather_dim
        self.n_past_years = n_past_years

        # Calculate max sequence length (same as other models)
        self.max_len = (n_past_years + 1) * 52

        # Single linear layer: flattened weather + soil + past yields -> yield prediction
        # weather: max_len * weather_dim
        # soil: 11 measurements * 6 depths = 66 (constant across years, only 1 set)
        # past yields: n_past_years + 1
        soil_dim = 11 * 6  # 11 measurements * 6 depths
        input_dim = self.max_len * weather_dim + soil_dim + n_past_years + 1
        self.linear = nn.Linear(input_dim, 1)

    def load_pretrained(self, pretrained_model: Optional["BaseModel"]):
        """
        Load pretrained model parameters. For linear model, this is optional.
        """
        if pretrained_model is not None:
            self.logger.info(
                f"Loading pretrained linear model: {pretrained_model.name}"
            )
            self.load_state_dict(pretrained_model.state_dict())
        else:
            self.logger.info("No pretrained model provided for linear model")

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        y_past,
        soil,
    ):
        """
        Forward pass through the linear model.

        Args:
            padded_weather: Weather data [batch_size, seq_len, weather_dim]
            coord: Coordinates [batch_size, 2]
            year: Year information [batch_size, seq_len]
            interval: Interval information [batch_size, 1]
            weather_feature_mask: Mask for weather features [batch_size, seq_len, weather_dim]
            y_past: Past yield values [batch_size, n_past_years + 1]
            soil: Soil features [batch_size, n_years, soil_dim]

        Returns:
            Predicted yield [batch_size, 1]
        """
        batch_size = padded_weather.shape[0]

        # Handle missing weather data by masking out missing values
        if weather_feature_mask is not None:
            weather_processed = padded_weather * (~weather_feature_mask)
        else:
            weather_processed = padded_weather

        # Flatten weather features: [batch_size, seq_len * weather_dim]
        weather_flattened = weather_processed.view(batch_size, -1)

        # Soil is constant across years, take first year only: [batch_size, 11, 6] -> [batch_size, 66]
        soil_flattened = soil[:, 0, :, :].reshape(batch_size, -1)

        # Concatenate flattened weather, soil, and past yields
        linear_input = torch.cat([weather_flattened, soil_flattened, y_past], dim=1)

        # Single linear layer prediction
        output = self.linear(linear_input)

        return output

    def compute_l2_regularization(self):
        """
        Compute L2 regularization term for ridge regression.
        Returns the sum of squared weights of the linear layer.
        """
        return torch.sum(self.linear.weight**2)
