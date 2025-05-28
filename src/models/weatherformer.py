import torch
import torch.nn as nn
from typing import Optional
from src.models.weatherbert import WeatherBERT
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE

"""
This class implements the WeatherFormer model.

The output is a tuple of mu, sigma, where mu is the mean and sigma is the standard deviation of the normal distribution.
"""


class WeatherFormer(WeatherBERT):
    def __init__(
        self,
        weather_dim,
        output_dim,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
        device=DEVICE,
    ):
        super(WeatherFormer, self).__init__(
            weather_dim=weather_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim_factor=hidden_dim_factor,
            max_len=max_len,
            device=device,
        )

        # Override the output projection to be 2x the size for mu and sigma
        hidden_dim = hidden_dim_factor * num_heads
        self.out_proj = nn.Linear(hidden_dim, 2 * output_dim)

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        temporal_index: torch.Tensor,
        weather_feature_mask: Optional[
            torch.Tensor
        ] = None,  # batch_size x seq_len x n_features,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2
        temporal_index: batch_size x 2
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len
        """
        # Call parent WeatherBERT forward method
        output = super().forward(
            weather=weather,
            coords=coords,
            temporal_index=temporal_index,
            weather_feature_mask=weather_feature_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Split output into mu and log_var (VAE-style parameterization)
        mu = output[..., : self.output_dim]
        log_var = output[..., self.output_dim :]

        # Compute sigma from log variance: sigma = exp(0.5 * log_var)
        sigma = torch.exp(0.5 * log_var)

        # Clip sigma to prevent numerical instability
        sigma = torch.clamp(sigma, min=1e-6)

        return mu, sigma
