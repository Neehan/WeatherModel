import torch
import torch.nn as nn
from typing import Optional
from src.pretraining.models.weatherbert import WeatherBERT
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE

"""
This class implements the WeatherFormer model.

The output is a tuple of mu_x, sigma, where mu_x is the mean and sigma is the standard deviation of the normal distribution.
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
        k=8,
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
        # override the name
        self.name = "weatherformer"
        self.k = k

        # Override the output projection to be 2x the size for mu_x and sigma
        hidden_dim = hidden_dim_factor * num_heads
        self.out_proj = nn.Linear(hidden_dim, 2 * output_dim)

        # Initialize mixture of Gaussians prior parameters (K components)
        self.mu_k = nn.Parameter(torch.zeros(k, max_len, output_dim))
        self.log_var_k = nn.Parameter(torch.zeros(k, max_len, output_dim))

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
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
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Split output into mu_x and log_var (VAE-style parameterization)
        mu_x = output[..., : self.output_dim]
        log_var = output[..., self.output_dim :]

        # Compute sigma from log variance: sigma = exp(0.5 * log_var)
        var_x = torch.exp(log_var)

        # Clip sigma to prevent numerical instability
        var_x = torch.clamp(var_x, min=1e-8, max=25)  # sigma^2 is in [1e-8, 25]

        return mu_x, var_x
