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
        mlp_input_dim: int,
        device: torch.device,
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        **model_size_params,
    ):
        # Call parent init but override the weather model
        super().__init__(
            name, mlp_input_dim, device, weather_dim, output_dim, **model_size_params
        )
        self.name = "weatherautoencoder_sine_yield"

        # Replace the WeatherBERT with WeatherAutoencoder
        self.weather_model = WeatherAutoencoder(
            weather_dim=weather_dim,
            output_dim=output_dim,
            device=device,
            **model_size_params,
        )

        # p(z) ~ N(A_p * sin(theta * z), sigma^2_p)
        self.positions = (
            torch.arange(self.weather_model.max_len, dtype=torch.float, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        self.theta_p = nn.Linear(1, output_dim)
        self.A_p = nn.Parameter(
            torch.randn(1, self.weather_model.max_len, output_dim) * 0.1
        )
        self.log_var_p = nn.Parameter(
            torch.randn(1, self.weather_model.max_len, output_dim) * 0.1 - 1
        )

        self.log_var_x = nn.Sequential(
            nn.Linear(weather_dim, 2 * weather_dim),
            nn.GELU(),
            nn.Linear(2 * weather_dim, output_dim),
        )

    def forward(self, input_data):
        # Prepare weather input using inherited method
        padded_weather, coord, year, interval, weather_feature_mask = input_data

        batch_size, seq_len = padded_weather.shape[:2]

        # WeatherFormerMixture expects individual arguments, not a tuple
        # and returns (mu_x, var_x, mu_k, var_k) instead of just weather embeddings
        mu_x = self.weather_model(
            padded_weather,
            coord,
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
        )
        mu_x = self._impute_weather(padded_weather, mu_x, weather_feature_mask)
        # batch size x seq_len x (2 n features)
        log_var_x = self.log_var_x(mu_x)
        var_x = torch.exp(log_var_x)

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon
        z = self._impute_weather(padded_weather, z, weather_feature_mask)
        # Compute sinusoidal prior: p(z) ~ N(A_p * sin(theta * pos * period), sigma^2_p)
        # period: (batch_size, 1, 1)

        # Slice parameters to match sequence length before computation
        positions_seq = self.positions[:, :seq_len, :]  # (1, seq_len, 1)
        theta_p_seq = self.theta_p(positions_seq)  # (batch_size, seq_len, output_dim)
        A_p_seq = self.A_p[:, :seq_len, :]  # (1, seq_len, output_dim)
        log_var_p_seq = self.log_var_p[:, :seq_len, :]  # (1, seq_len, output_dim)

        # Compute mean and variance of prior: (batch_size, seq_len, output_dim)
        mu_p = A_p_seq * torch.sin(theta_p_seq)
        var_p = torch.exp(log_var_p_seq)

        # Flatten the weather representation for MLP
        weather_repr_flat = z.reshape(z.size(0), -1)

        # Clamp variances for numerical stability before returning
        var_x = torch.clamp(var_x, min=1e-8, max=1)
        var_p = torch.clamp(var_p, min=1e-8, max=1)

        # Pass through MLP to get yield prediction
        yield_pred = self.mlp(weather_repr_flat)
        return yield_pred, z, mu_x, var_x, mu_p, var_p
