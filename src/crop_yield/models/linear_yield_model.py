import torch
import torch.nn as nn
from typing import Optional

from src.base_models.base_model import BaseModel


class LinearYieldModel(BaseModel):
    """
    USDA-style linear regression model following Westcott & Jewison (2013).

    Corn features:
    - Trend (year)
    - July average temperature
    - July precipitation
    - July precipitation squared
    - June precipitation shortfall
    - Average of past years yields

    Soybean features:
    - Trend (year)
    - July-August average temperature
    - July-August precipitation
    - July-August precipitation squared
    - June precipitation shortfall
    - Average of past years yields
    """

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
        self.weather_dim = weather_dim
        self.n_past_years = n_past_years
        self.crop_type = crop_type

        # USDA features: trend, temp, precip, precip^2, June shortfall, avg past yield
        input_dim = 6
        self.linear = nn.Linear(input_dim, 1)

        self.june_shortfall_percentile = 0.10

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

    def _extract_usda_features(self, padded_weather, year, y_past):
        """
        Extract USDA-style features from raw weather data.

        Weather indices: 1=max_temp, 2=min_temp, 7=precip
        """
        n_weeks_per_year = 52
        current_year_weather = padded_weather[:, -n_weeks_per_year:, :]

        max_temp = current_year_weather[:, :, 1]
        min_temp = current_year_weather[:, :, 2]
        precip = current_year_weather[:, :, 7]
        avg_temp = (max_temp + min_temp) / 2.0

        # Normalize year by subtracting base year (e.g., 1980)
        trend = (torch.floor(year[:, -1]) - 1970.0).unsqueeze(1) / 100.0

        june_weeks = slice(22, 26)

        # Corn: July only (weeks 26-30)
        # Soybean: July-August (weeks 26-34)
        if self.crop_type == "corn":
            critical_weeks = slice(26, 30)
        else:
            critical_weeks = slice(26, 34)

        critical_temp = avg_temp[:, critical_weeks].mean(dim=1, keepdim=True)
        critical_precip = precip[:, critical_weeks].mean(dim=1, keepdim=True)
        # Preserve sign with abs(x)*x for asymmetric effect
        critical_precip_sq = torch.abs(critical_precip) * critical_precip

        june_precip = precip[:, june_weeks].mean(dim=1, keepdim=True)
        june_threshold = torch.quantile(
            june_precip.view(-1), self.june_shortfall_percentile
        )

        june_shortfall = torch.where(
            june_precip < june_threshold,
            june_threshold - june_precip,
            torch.zeros_like(june_precip),
        )

        avg_past_yield = y_past[:, -5:-1].mean(dim=1, keepdim=True)

        features = torch.cat(
            [
                trend,
                critical_temp,
                critical_precip,
                critical_precip_sq,
                june_shortfall,
                avg_past_yield,
            ],
            dim=1,
        )

        return features

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
        Forward pass through USDA-style linear model.

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
        features = self._extract_usda_features(padded_weather, year, y_past)
        output = self.linear(features)
        return output
