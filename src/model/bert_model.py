import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
from typing import Optional
from .model import WFPositionalEncoding


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

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.input_scaler = nn.Embedding(
            num_embeddings=MAX_GRANULARITY_DAYS, embedding_dim=input_dim, padding_idx=0
        )
        torch.nn.init.constant_(self.input_scaler.weight.data, 1.0)

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
        weather,
        coords,
        temporal_index,
        weather_feature_mask=None,  # batch_size x seq_len x n_features,
        src_key_padding_mask=None,  # batch_size x seq_len
    ):

        batch_size, seq_len, n_features = weather.shape

        # temporal index is index in time and temporal granularity ()
        temporal_granularity = temporal_index[:, 1].int()
        temp_embedding = self.input_scaler(temporal_granularity)

        # mask the masked dimensions and scale the rest
        weather = weather * temp_embedding.view(batch_size, 1, n_features)
        if weather_feature_mask is not None:
            weather = weather * (~weather_feature_mask)

            # scalers for for masked dimensions = true becomes zero
            # temp_embedding = (~weather_feature_mask).unsqueeze(0) * temp_embedding

        weather = self.in_proj(weather)
        # add temporal positional encoding
        # weather += self.temporal_pos_encoding(temporal_granularity).unsqueeze(1)

        weather = self.positional_encoding(weather, coords)
        weather = self.transformer_encoder(
            weather, src_key_padding_mask=src_key_padding_mask
        )
        weather = self.out_proj(weather)

        return weather
