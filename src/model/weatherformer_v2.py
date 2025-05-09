import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
from typing import Optional


class WFPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
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

    def forward(self, token_embedding, coords):
        """
        Forward method for adding positional encoding.

        Args:
        token_embedding: Tensor, shape [batch_size, seq_len, d_model]
        latitude: Tensor, shape [batch_size, 1]
        longitude: Tensor, shape [batch_size, 1]

        Returns:
        Tensor with positional encoding added, same shape as x.
        """
        device = coords.device
        div_term = self.div_term.to(device)
        position_list = self.position_list.to(device)

        batch_size, seq_len, d_model = token_embedding.shape
        latitude, longitude = coords[:, :1], coords[:, 1:]
        # Normalize latitude and longitude
        lat_norm = (latitude / 180.0) * math.pi
        lon_norm = (longitude / 180.0) * math.pi

        # Create geo encoding
        custom_pe = torch.zeros(batch_size, seq_len, d_model, device=device)

        # geo pe
        custom_pe[:, :, 2::4] = torch.sin(lat_norm * div_term).unsqueeze(1)
        custom_pe[:, :, 3::4] = torch.cos(lon_norm * div_term).unsqueeze(1)

        time_frequency = (position_list[:seq_len, :] * div_term).unsqueeze(0)
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
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=CONTEXT_LENGTH,
        device=DEVICE,
    ):
        super(Weatherformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.input_scaler = nn.Embedding(
            num_embeddings=MAX_GRANULARITY_DAYS, embedding_dim=input_dim, padding_idx=0
        )
        # torch.nn.init.constant_(self.input_scaler.weight.data, 1.0)

        # self.temporal_pos_encoding = nn.Embedding(
        #     num_embeddings=MAX_GRANULARITY_DAYS, embedding_dim=hidden_dim, padding_idx=0
        # )

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = WFPositionalEncoding(
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
        data,
        weather_feature_mask=None,  # n_features,
        src_key_padding_mask=None,  # batch_size x seq_len
    ):

        weather, coords, temporal_index = data
        batch_size, seq_len, n_features = weather.shape

        # temporal index is index in time and temporal granularity ()
        temporal_granularity = temporal_index[:, 1].int()
        temp_embedding = self.input_scaler(temporal_granularity)

        # add timescale embedding (basically weekly, monthly etc)
        weather += temp_embedding.view(batch_size, 1, n_features)
        if weather_feature_mask is not None:
            # if the shape is batch_size x weather indices, keep the first copy
            if len(weather_feature_mask.shape) > 1:
                weather_feature_mask = weather_feature_mask[0, :]
            weather[:, :, weather_feature_mask] = 0

            # scalers for for masked dimensions = true becomes zero
            # temp_embedding = (~weather_feature_mask).unsqueeze(0) * temp_embedding

        weather = self.in_proj(weather)
        weather = self.positional_encoding(weather, coords)
        weather = self.transformer_encoder(
            weather, src_key_padding_mask=src_key_padding_mask
        )
        weather = self.out_proj(weather)

        return weather
