import torch
import torch.nn as nn

from src.crop_yield.models.weatherformer_yield_model import WeatherFormerYieldModel
from src.pretraining.models.weatherformer import WeatherFormer


class DecoderYieldModel(WeatherFormerYieldModel):
    """
    Decoder-based yield prediction model that adds weather reconstruction capability.

    This model extends WeatherFormerYieldModel to include an MLP decoder that reconstructs
    weather from the latent representations, enabling reconstruction loss computation.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        # Call parent init
        super().__init__(name, device, weather_dim, n_past_years, **model_size_params)

        # Add MLP decoder for weather reconstruction
        # Simple 2-layer MLP: weather_dim -> hidden -> weather_dim
        hidden_dim = 128
        self.weather_decoder = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, weather_dim),
        )

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        y_past,
    ):
        # WeatherFormer expects individual arguments, not a tuple
        # and returns (mu, sigma) instead of just weather embeddings
        mu_x, var_x = self.weather_model(
            padded_weather,
            coord,
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
        )

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon

        # Decode weather from z before imputation
        weather_pred = self.weather_decoder(z)

        # Impute weather for yield prediction
        z_imputed = self._impute_weather(padded_weather, z, weather_feature_mask)

        # Compute yield prediction using imputed weather
        yield_pred = self.yield_model(
            z_imputed,
            coord,
            year,
            interval,
            weather_feature_mask=None,
            y_past=y_past,
        )
        return yield_pred, z, mu_x, var_x, weather_pred
