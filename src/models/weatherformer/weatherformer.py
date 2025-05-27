import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.weatherbert.vanilla_pos_encoding import VanillaPositionalEncoding
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE

"""
This class implements the WeatherFormer model.

The output is a tuple of mu, sigma, where mu is the mean and sigma is the standard deviation of the normal distribution.
"""


class WeatherFormer(nn.Module):
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
        super(WeatherFormer, self).__init__()

        self.weather_dim = weather_dim
        self.input_dim = (
            weather_dim + 2 + 1
        )  # weather (normalized) + coords (/360) + temporal_granularity in days / 30
        self.output_dim = output_dim
        self.max_len = max_len

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.in_proj = nn.Linear(self.input_dim, hidden_dim)
        self.positional_encoding = VanillaPositionalEncoding(
            hidden_dim, max_len=max_len, device=device
        )
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            device=device,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

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
        batch_size, seq_len, n_features = weather.shape

        if n_features != self.weather_dim:
            raise ValueError(
                f"expected {self.weather_dim} weather features but received {n_features} features"
            )

        # temporal index is index in time and temporal granularity (in days)
        temporal_granularity = temporal_index[:, 1]

        # Expand temporal_granularity to match weather and coords dimensions
        # From [batch_size] to [batch_size, seq_len, 1]
        temporal_granularity = (
            temporal_granularity.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, seq_len, 1)
        )

        # Expand coords to match sequence length if needed
        coords = coords.unsqueeze(1).expand(batch_size, seq_len, 2)

        # mask the masked dimensions
        if weather_feature_mask is not None:
            weather = weather * (~weather_feature_mask)

        input_tensor = torch.cat([weather, coords, temporal_granularity], dim=2)
        input_tensor = self.in_proj(input_tensor)
        input_tensor = self.positional_encoding(input_tensor)
        input_tensor = self.transformer_encoder(
            input_tensor, src_key_padding_mask=src_key_padding_mask
        )
        output = self.out_proj(input_tensor)

        # Split output into mu and log_var (VAE-style parameterization)
        mu = output[..., : self.output_dim]
        log_var = output[..., self.output_dim :]

        # Compute sigma from log variance: sigma = exp(0.5 * log_var)
        sigma = torch.exp(0.5 * log_var)

        return mu, sigma
