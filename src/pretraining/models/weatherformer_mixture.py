import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.pretraining.models.weatherformer import WeatherFormer
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE

"""
This class implements the WeatherFormerMixture model, which extends WeatherFormer
with mixture parameters mu_k and var_k.
"""


class WeatherFormerMixture(WeatherFormer):
    def __init__(
        self,
        weather_dim,
        output_dim,
        k=7,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
        device=DEVICE,
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

        # Mixture parameters: mu_k and log_var_k of shape (k, max_len, weather_dim)
        # Initialize mixture means with small random values instead of zeros
        self.mu_k = nn.Parameter(torch.randn(k, max_len, output_dim) * 0.1)
        # Initialize log_var_k to give var_k around 0.1-1.0 range instead of exactly 1.0
        self.log_var_k = nn.Parameter(torch.randn(k, max_len, output_dim) * 0.1 - 1.0)

    def load_pretrained(self, pretrained_model: "WeatherFormerMixture"):
        # super().load_pretrained(pretrained_model)
        self.in_proj = pretrained_model.in_proj
        self.positional_encoding = pretrained_model.positional_encoding
        self.transformer_encoder = pretrained_model.transformer_encoder
        # self.out_proj = pretrained_model.out_proj
        self.mu_k = pretrained_model.mu_k
        self.log_var_k = pretrained_model.log_var_k

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2
        year: batch_size x 2
        interval: batch_size x 2
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len

        Returns:
            mu_x: batch_size x seq_len x n_features
            var_x: batch_size x seq_len x n_features
            mu_k: k x seq_len x n_features
            var_k: k x seq_len x n_features
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

        # Truncate mixture parameters to match runtime sequence length
        mu_k = self.mu_k[:, :seq_len, :]  # k x seq_len x n_features
        log_var_k = self.log_var_k[:, :seq_len, :]  # k x seq_len x n_features

        # Compute var_k from log_var_k
        var_k = torch.exp(log_var_k)

        # Clamp var_k to prevent numerical instability
        var_k = torch.clamp(var_k, min=0.01, max=1)  # var_k is in [0.01, 1]

        return mu_x, var_x, mu_k, var_k
