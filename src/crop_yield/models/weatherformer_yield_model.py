import torch
import torch.nn as nn

from src.crop_yield.models.weatherbert_yield_model import WeatherBERTYieldModel
from src.pretraining.models.weatherformer import WeatherFormer
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS


class WeatherFormerYieldModel(WeatherBERTYieldModel):
    """
    WeatherFormer-based yield prediction model that handles probabilistic weather representations.

    This model extends WeatherBERTYieldModel to work with WeatherFormer's (mu, sigma) output
    and uses the reparameterization trick for sampling weather representations.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        # Call parent init but override the weather model
        super().__init__(name, device, weather_dim, n_past_years, **model_size_params)

        # Replace the WeatherBERT with WeatherFormer
        self.weather_model = WeatherFormer(
            weather_dim=weather_dim,
            output_dim=weather_dim,
            device=device,
            **model_size_params,
        )

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        practices,
        soil,
        y_past,
    ):
        # Prepare weather input using inherited method

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

        z = self._impute_weather(padded_weather, z, weather_feature_mask)

        # we imputed weather, the mask is not necessary
        yield_pred = self.yield_model(
            z,
            coord,
            year,
            interval,
            weather_feature_mask=None,
            y_past=y_past,
        )
        return yield_pred, mu_x, var_x
