import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        assert (
            d_model % 2 == 0
        ), "d_model should be divisible by 4 for separate encoding"

        # we did seq_len + 1 for max len cause there's a summary vector
        super(PositionalEncoding, self).__init__()

        # Create a position array (time encoding)
        self.position_list = (
            torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        )

        # max is 10k**(-1/2), cause we have the bias
        self.div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ).to(device)

    def forward(self, token_embedding):
        """
        Forward method for adding positional encoding.

        Args:
        token_embedding: Tensor, shape [batch_size, seq_len, d_model]
        latitude: Tensor, shape [batch_size, 1]
        longitude: Tensor, shape [batch_size, 1]

        Returns:
        Tensor with positional encoding added, same shape as x.
        """
        device = token_embedding.device
        div_term = self.div_term.to(device)
        position_list = self.position_list.to(device)

        batch_size, seq_len, d_model = token_embedding.shape

        # Create geo encoding
        custom_pe = torch.zeros(batch_size, seq_len, d_model, device=device)

        time_frequency = (position_list[:seq_len, :] * div_term).unsqueeze(0)
        # encode time in 2k and 2k + 1
        custom_pe[:, :, 0::2] = torch.sin(time_frequency)
        custom_pe[:, :, 1::2] = torch.cos(time_frequency)

        # Add positional encoding to input
        token_embedding += custom_pe
        return token_embedding


class WeatherBERT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=CONTEXT_LENGTH,
        device=DEVICE,
    ):
        super(WeatherBERT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.mask_value = nn.Parameter(torch.tensor([0.0], device=device))

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(
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

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        weather,
        coords,
        temporal_index,
        weather_feature_mask=None,  # batch_size x seq_len,
        src_key_padding_mask=None,  # batch_size x seq_len
    ):

        batch_size, seq_len, n_features = weather.size()
        if weather_feature_mask is not None:
            # invert cause mask = true means we want it to be masked
            weather = weather * (~weather_feature_mask)

        weather = self.in_proj(weather)
        weather = self.positional_encoding(weather)
        weather = self.transformer_encoder(
            weather, src_key_padding_mask=src_key_padding_mask
        )
        weather = self.out_proj(weather)

        return weather
