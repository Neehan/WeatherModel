import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
from typing import Optional


class WFPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device=DEVICE):
        assert (
            d_model % 4 == 0
        ), "d_model should be divisible by 4 for separate encoding"

        # we did seq_len + 1 for max len cause there's a summary vector
        super(WFPositionalEncoding, self).__init__()

        # Create a position array (time encoding)
        self.position_list = (
            torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        )

        # max is 10k**(-1/2), cause we have the bias
        self.div_term = torch.exp(
            torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model)
        ).to(device)

    def forward(self, token_embedding, coords, device=DEVICE):
        """
        Forward method for adding positional encoding.

        Args:
        token_embedding: Tensor, shape [batch_size, seq_len, d_model]
        latitude: Tensor, shape [batch_size, 1]
        longitude: Tensor, shape [batch_size, 1]

        Returns:
        Tensor with positional encoding added, same shape as x.
        """
        batch_size, seq_len, d_model = token_embedding.shape
        latitude, longitude = coords[:, :1], coords[:, 1:]
        # Normalize latitude and longitude
        lat_norm = (latitude / 180.0) * math.pi
        lon_norm = (longitude / 180.0) * math.pi

        # Create geo encoding
        custom_pe = torch.zeros(batch_size, seq_len, d_model, device=device)

        # geo pe
        custom_pe[:, :, 2::4] = torch.sin(lat_norm * self.div_term).unsqueeze(1)
        custom_pe[:, :, 3::4] = torch.cos(lon_norm * self.div_term).unsqueeze(1)

        time_frequency = (self.position_list * self.div_term).unsqueeze(0)
        # encode time in 4k and 4k + 1
        custom_pe[:, :, 0::4] = torch.sin(time_frequency)
        custom_pe[:, :, 1::4] = torch.cos(time_frequency)

        # Add positional encoding to input
        token_embedding += custom_pe
        return token_embedding


class Weatherformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=8,
        num_layers=3,
        hidden_dim_factor=8,
        max_len=CONTEXT_LENGTH,
    ):
        super(Weatherformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        # allow each weather feature to be scaled based on input mask
        # up to 30 days + 1 for padding, granuality changes the scaling
        self.input_scaler = nn.Embedding(num_embeddings=31, embedding_dim=input_dim)

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = WFPositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            device=DEVICE,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        weather,
        coords,
        temporal_index,
        weather_feature_mask=None,
        src_key_padding_mask=None,
    ):

        # temporal index is index in time and temporal granularity ()
        temporal_granularity = temporal_index[:, 1:].unsqueeze(2).int()
        temp_embedding = self.input_scaler(temporal_granularity)

        # mask certain features in the input weather
        if weather_feature_mask is not None:
            # scaler for masked dimensions = true becomes zero
            input_mask = (~weather_feature_mask).unsqueeze(0) * temp_embedding

            # mask the masked dimensions and scale the rest
            weather = weather * input_mask.view(weather.shape[0], 1, -1)

        weather = self.input_proj(weather)
        weather = self.positional_encoding(weather, coords)
        weather = self.transformer_encoder(
            weather, src_key_padding_mask=src_key_padding_mask
        )
        weather = self.fc(weather)
        return weather
