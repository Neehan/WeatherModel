import torch
import torch.nn as nn
from src.pretraining.models.weatherautoencoder import WeatherAutoencoder
from src.crop_yield.models.weatherautoencoder_yield_model import (
    WeatherAutoencoderYieldModel,
)
from src.utils.constants import TOTAL_WEATHER_VARS


class WeatherAutoencoderSineYieldModel(WeatherAutoencoderYieldModel):
    """
    WeatherAutoencoderSine-based yield prediction model that handles probabilistic weather representations
    with sinusoidal priors.

    This model extends WeatherBERTYieldModel to work with WeatherAutoencoder's (mu_x, var_x, mu_p, var_p) output
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
        self.name = "weatherautoencoder_sine_yield"

        # p(z) ~ N(A_p * sin(theta * z), sigma^2_p)
        max_len = self.yield_model.max_len
        self.positions = (
            torch.arange(max_len, dtype=torch.float, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        self.theta_p = nn.Linear(1, weather_dim)
        self.A_p = nn.Parameter(
            torch.randn(1, max_len, weather_dim, device=device) * 0.1
        )
        self.log_var_p = nn.Parameter(
            torch.randn(1, max_len, weather_dim, device=device) * 0.1 - 1
        )

        self.log_var_x = nn.Sequential(
            nn.Linear(weather_dim, 4 * weather_dim),
            nn.GELU(),
            nn.Linear(4 * weather_dim, weather_dim),
        )

    def _compute_sinusoidal_prior(self):
        """
        Compute sinusoidal prior parameters: p(z) ~ N(A_p * sin(theta * pos), sigma^2_p)

        Args:
            seq_len: Sequence length to slice parameters to

        Returns:
            tuple: (mu_p, var_p) - mean and variance of the sinusoidal prior
        """

        # Compute mean and variance of prior: (batch_size, seq_len, output_dim)
        mu_p = self.A_p * torch.sin(self.theta_p(self.positions))
        var_p = torch.exp(self.log_var_p)

        return mu_p, var_p

    def forward(
        self,
        padded_weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        y_past,
    ):
        # Compute sinusoidal prior: p(z) ~ N(A_p * sin(theta * pos * period), sigma^2_p)
        mu_p, var_p = self._compute_sinusoidal_prior()

        # WeatherFormerMixture expects individual arguments, not a tuple
        # and returns (mu_x, var_x, mu_k, var_k) instead of just weather embeddings
        mu_x = self.weather_model(
            padded_weather,
            coord,
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
        )
        # predict missing only and keep original weather for rest
        mu_x = self._impute_weather(padded_weather, mu_x, weather_feature_mask)
        log_var_x = self.log_var_x(mu_x)
        var_x = torch.exp(log_var_x)

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon
        # predict missing only and keep original weather for rest
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

        # Clamp variances for numerical stability before returning
        var_x = torch.clamp(var_x, min=1e-8, max=1)
        var_p = torch.clamp(var_p, min=1e-8, max=1)

        return yield_pred, z, mu_x, var_x, mu_p, var_p
