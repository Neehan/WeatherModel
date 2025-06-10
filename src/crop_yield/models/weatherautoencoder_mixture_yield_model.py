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
        max_len: int,
        **model_size_params,
    ):
        # Call parent init but override the weather model
        super().__init__(
            name, device, weather_dim, n_past_years, max_len, **model_size_params
        )
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
            nn.Linear(2 * weather_dim, 4 * weather_dim),
            nn.GELU(),
            nn.Linear(4 * weather_dim, weather_dim),
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

        seq_len = padded_weather.shape[1]

        # WeatherFormerMixture expects individual arguments, not a tuple
        # and returns (mu_x, var_x, mu_k, var_k) instead of just weather embeddings
        mu_x = self.weather_model(
            padded_weather,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
        )
        # batch size x seq_len x (2 n features)
        log_var_x = self.log_var_x(torch.cat([mu_x, padded_weather], dim=2))
        var_x = torch.exp(log_var_x)

        mu_k = self.mu_k[:, :seq_len, :]
        var_k = torch.exp(self.log_var_k[:, :seq_len, :])

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon
        final_weather = self._impute_weather(padded_weather, z, weather_feature_mask)

        # Clamp variances for numerical stability before returning
        var_x = torch.clamp(var_x, min=1e-8, max=1)
        var_k = torch.clamp(var_k, min=1e-8, max=1)

        # Pass through MLP to get yield prediction
        yield_pred = self.yield_model(
            final_weather,
            coord,
            year,
            interval,
            weather_feature_mask,
            practices,
            soil,
            y_past,
        )
        return yield_pred, z, mu_x, var_x, mu_k, var_k
