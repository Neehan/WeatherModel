import torch
import torch.nn as nn
from src.pretraining.models.weatherautoencoder import WeatherAutoencoder
from src.crop_yield.models.weatherautoencoder_yield_model import (
    WeatherAutoencoderYieldModel,
)
from src.utils.constants import TOTAL_WEATHER_VARS


class WeatherAutoencoderMixtureYieldModel(WeatherAutoencoderYieldModel):
    """
    WeatherAutoencoderMixture-based yield prediction model that handles probabilistic weather representations
    with Gaussian mixture priors.

    This model extends WeatherBERTYieldModel to work with WeatherFormerMixture's (mu_x, var_x, mu_k, var_k) output
    and uses the reparameterization trick for sampling weather representations.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        k: int,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        # Call parent init but override the weather model
        super().__init__(name, device, weather_dim, n_past_years, **model_size_params)
        self.name = "weatherautoencoder_mixture_yield"

        # Replace the WeatherBERT with WeatherAutoencoder
        self.weather_model = WeatherAutoencoder(
            weather_dim=weather_dim,
            output_dim=weather_dim,
            device=device,
            **model_size_params,
        )
        self.k = k
        # Initialize mu_k to give var_k around (-0.2, 0.2) range
        self.mu_k = nn.Parameter(
            torch.randn(k, self.weather_model.max_len, weather_dim) * 0.1
        )
        # Initialize log_var_k to give var_k around (-1.2, -0.8) range
        self.log_var_k = nn.Parameter(
            torch.randn(k, self.weather_model.max_len, weather_dim) * 0.1 - 1.0
        )
        self.log_var_x = nn.Sequential(
            nn.Linear(weather_dim, 4 * weather_dim),
            nn.GELU(),
            nn.Linear(4 * weather_dim, weather_dim),
        )

    def forward(self, weather, coord, year, interval, weather_feature_mask):
        # Prepare weather input using inherited method

        seq_len = weather.shape[1]

        # WeatherFormerMixture expects individual arguments, not a tuple
        # and returns (mu_x, var_x, mu_k, var_k) instead of just weather embeddings
        mu_x = self.weather_model(
            weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )
        # keep original weather and predict only missing ones
        mu_x = self._impute_weather(weather, mu_x, weather_feature_mask)
        log_var_x = self.log_var_x(mu_x)
        var_x = torch.exp(log_var_x)

        mu_k = self.mu_k[:, :seq_len, :]
        var_k = torch.exp(self.log_var_k[:, :seq_len, :])

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon
        # keep original weather and predict only missing ones
        z = self._impute_weather(weather, z, weather_feature_mask)

        # Clamp variances for numerical stability before returning
        var_x = torch.clamp(var_x, min=1e-8, max=1)
        var_k = torch.clamp(var_k, min=1e-8, max=1)

        # Pass through MLP to get yield prediction
        yield_pred = self.yield_model(
            z,
            coord,
            year,
            interval,
            weather_feature_mask,
        )
        return yield_pred, z, mu_x, var_x, mu_k, var_k
