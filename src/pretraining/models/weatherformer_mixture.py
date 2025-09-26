import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.pretraining.models.weatherformer import WeatherFormer
from src.utils.constants import MAX_CONTEXT_LENGTH

"""
This class implements the WeatherFormerMixture model, which extends WeatherFormer
with mixture parameters mu_k and var_k. The mu_k is computed using sinusoidal functions
while var_k remains learnable.
"""


class WeatherFormerMixture(WeatherFormer):
    def __init__(
        self,
        weather_dim,
        output_dim,
        device,
        k=7,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
    ):
        super(WeatherFormerMixture, self).__init__(
            weather_dim=weather_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim_factor=hidden_dim_factor,
            max_len=max_len,
            device=device,
        )
        # override the name
        self.name = "weatherformer_mixture"
        self.k = k

        # Position tensor for sinusoidal computation
        self.positions = torch.arange(max_len, dtype=torch.float, device=device).view(
            1, 1, max_len, 1
        )

        # Sinusoidal parameters for mixture means mu_k
        # Shape: (1, k, max_len, output_dim) to avoid unsqueezing later
        self.frequency = nn.Parameter(torch.randn(1, k, max_len, output_dim) * 0.1)
        self.phase = nn.Parameter(torch.randn(1, k, max_len, output_dim) * 0.1)
        self.amplitude = nn.Parameter(torch.randn(1, k, max_len, output_dim) * 0.1)

        # Mixture variances: log_var_k of shape (k, max_len, output_dim)
        # Initialize log_var_k to give var_k around 0.1-1.0 range instead of exactly 1.0
        self.log_var_k = nn.Parameter(
            torch.randn(1, k, max_len, output_dim) * 0.1 - 1.0
        )

        # Learnable mixture weights (log-probabilities)
        # Shape: (1, k) - one weight per mixture component
        # Initialize to uniform weights: log(1/k) = -log(k)
        uniform_log_weight = -torch.log(torch.tensor(k, dtype=torch.float32)).item()
        self.mixture_logits = nn.Parameter(torch.full((1, k), uniform_log_weight))

    def load_pretrained(
        self, pretrained_model: "WeatherFormerMixture", load_out_proj=True
    ):
        if self.k != pretrained_model.k:
            raise ValueError(
                f"k mismatch: {self.k} != {pretrained_model.k}. Please ensure the models are compatible."
            )
        super().load_pretrained(pretrained_model, load_out_proj)
        if load_out_proj:
            self.frequency = copy.deepcopy(pretrained_model.frequency)
            self.phase = copy.deepcopy(pretrained_model.phase)
            self.amplitude = copy.deepcopy(pretrained_model.amplitude)
            self.log_var_k = copy.deepcopy(pretrained_model.log_var_k)
            self.mixture_logits = copy.deepcopy(pretrained_model.mixture_logits)
        self.k = pretrained_model.k

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2
        year: batch_size x seq_len
        interval: batch_size
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len

        Returns:
            mu_x: batch_size x seq_len x n_features
            var_x: batch_size x seq_len x n_features
            mu_k: batch_size x k x seq_len x n_features
            var_k: batch_size x k x seq_len x n_features
            mixture_weights: batch_size x k (log probabilities)
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

        # [1, k, seq_len, 1]
        pos = self.positions[:, :, :seq_len, :]
        amp = self.amplitude[:, :, :seq_len, :]
        freq = self.frequency[:, :, :seq_len, :]
        phase = self.phase[:, :, :seq_len, :]
        # [batch size, k, seq_len, 1]
        scaled_pos = pos * 2 * torch.pi * interval.view(batch_size, 1, 1, 1) / 365.0

        # Compute sinusoidal means for each mixture component
        # Broadcasting: (batch_size, k, seq_len, output_dim)
        mu_k = amp * torch.sin(freq * scaled_pos + phase)

        # Compute mixture variances from log_var_k
        var_k = torch.exp(self.log_var_k[:, :, :seq_len, :])

        # Clamp var_k to prevent numerical instability
        var_k = torch.clamp(var_k, min=1e-6, max=1)  # var_k is in [0.01, 1]

        # Expand var_k to include batch dimension
        var_k = var_k.expand(
            batch_size, -1, -1, -1
        )  # (batch_size, k, seq_len, output_dim)

        # Compute mixture weights (log probabilities)
        log_w_k = torch.log_softmax(self.mixture_logits, dim=1)  # (1,k,)
        # Expand to batch dimension
        log_w_k = log_w_k.expand(batch_size, -1)  # (batch_size, k)

        return mu_x, var_x, mu_k, var_k, log_w_k
