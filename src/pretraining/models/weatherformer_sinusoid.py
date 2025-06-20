import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.pretraining.models.weatherformer import WeatherFormer
from src.utils.constants import DEVICE, MAX_CONTEXT_LENGTH

"""
This class implements the WeatherFormerMixture model, which extends WeatherFormer
with mixture parameters mu_k and var_k.
"""


class WeatherFormerSinusoid(WeatherFormer):
    def __init__(
        self,
        weather_dim,
        output_dim,
        k=4,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
        device=DEVICE,
    ):
        super(WeatherFormerSinusoid, self).__init__(
            weather_dim=weather_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim_factor=hidden_dim_factor,
            max_len=max_len,
            device=device,
        )
        # override the name
        self.name = "weatherformer_sinusoid"
        self.positions = torch.arange(
            max_len, dtype=torch.float, device=device
        ).reshape(
            1, 1, max_len, 1
        )  #
        self.k = k

        # Initialize with shape (1, k, max_len, weather_dim) to avoid unsqueezing later
        self.frequency = nn.Parameter(torch.randn(1, k, max_len, weather_dim) * 0.1)
        self.phase = nn.Parameter(torch.randn(1, k, max_len, weather_dim) * 0.1)
        self.amplitude = nn.Parameter(torch.randn(1, k, max_len, weather_dim) * 0.1)
        self.log_var_prior = nn.Parameter(
            torch.randn(1, max_len, weather_dim) * 0.1 - 1
        )

    def load_pretrained(
        self, pretrained_model: "WeatherFormerSinusoid", load_out_proj=True
    ):
        super().load_pretrained(pretrained_model, load_out_proj)
        if self.k != pretrained_model.k:
            raise ValueError(
                f"k mismatch: {self.k} != {pretrained_model.k}. Please set k to the same value."
            )
        self.frequency = copy.deepcopy(pretrained_model.frequency)
        self.phase = copy.deepcopy(pretrained_model.phase)
        self.amplitude = copy.deepcopy(pretrained_model.amplitude)
        self.log_var_prior = copy.deepcopy(pretrained_model.log_var_prior)

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2
        year: batch_size x seq_len
        interval: batch_size x 1
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len

        Returns:
            mu_x: batch_size x seq_len x n_features
            var_x: batch_size x seq_len x n_features
            mu_p: batch_size x seq_len x n_features
            var_p: batch_size x seq_len x n_features
        """
        # Call parent WeatherFormer forward method
        mu_x, var_x = super().forward(
            weather=weather,
            coords=coords,
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Get the actual sequence length from the input
        seq_len = weather.shape[1]
        batch_size = weather.shape[0]

        # Compute sinusoidal prior: p(z) ~ N(A * sin(theta * pos + phase), sigma^2_p)
        # Parameters are already shaped as (1, k, max_len, weather_dim)
        amplitude = self.amplitude[:, :, :seq_len, :]  # (1, k, seq_len, weather_dim)
        phase = self.phase[:, :, :seq_len, :]  # (1, k, seq_len, weather_dim)
        frequency = self.frequency[:, :, :seq_len, :]  # (1, k, seq_len, weather_dim)

        # pos is (1, seq_len, 1)
        pos = self.positions[:, :, :seq_len, :]
        # scaled_pos is (1, 1, seq_len, 1) -> (batch_size, 1, seq_len, 1)
        scaled_pos = pos * 2 * torch.pi * interval.view(batch_size, 1, 1, 1) / 365.0

        # Now broadcasting works directly: (batch_size, k, seq_len, weather_dim)
        sines = amplitude * torch.sin(frequency * scaled_pos + phase)
        mu_p = torch.sum(
            sines, dim=1
        )  # sum over k dimension -> (batch_size, seq_len, weather_dim)
        var_p = torch.exp(self.log_var_prior)[:, :seq_len, :].expand(batch_size, -1, -1)

        # Clamp var_p to prevent numerical instability
        var_p = torch.clamp(var_p, min=1e-6, max=1)

        return mu_x, var_x, mu_p, var_p
