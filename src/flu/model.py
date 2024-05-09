from ..model.model import Weatherformer
from .constants import *

import torch
import torch.nn as nn
import copy
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()

        assert (
            dim_model % 2 == 0
        ), "dim_model should be divisible by 4 for separate encoding"
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model).to(DEVICE)
        positions_list = (
            torch.arange(0, max_len, dtype=torch.float).view(-1, 1).to(DEVICE)
        )  # 0, 1, 2, 3, 4, 5
        self.div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        ).to(
            DEVICE
        )  # 10000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/10000^(4i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * self.div_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(4i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * self.div_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Add positional encoding to input
        token_embedding += self.pos_encoding[: token_embedding.shape[1], :].unsqueeze(0)
        return token_embedding


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=8,
        num_layers=3,
        hidden_dim_factor=8,
    ):
        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.attn_layer = nn.Linear(hidden_dim, 1)  # Learnable attention layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_tensor, mask=None, return_sequence=False):
        embedded_tensor = self.embedding(input_tensor)
        encoded_tensor = self.positional_encoding(embedded_tensor)
        encoded_tensor = self.transformer_encoder(
            encoded_tensor, src_key_padding_mask=mask
        )
        # Compute attention weights
        attn_weights = self.attn_layer(encoded_tensor)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_summary = torch.sum(encoded_tensor * attn_weights, dim=1)
        weighted_summary = self.fc(weighted_summary)

        # Check if the full sequence should be returned
        if return_sequence:
            # Multiply entire encoded tensor with self.fc
            encoded_tensor = self.fc(encoded_tensor)
            return encoded_tensor, weighted_summary
        else:
            return weighted_summary


class FluPredictor(nn.Module):
    def __init__(
        self,
        pretrained_weatherformer: Weatherformer = None,
        hidden_dim=48,
        num_layers=3,
        **weatherformer_size_params
    ):
        super().__init__()

        self.weather_transformer = Weatherformer(
            len(WEATHER_PARAMS),
            hidden_dim,
            max_len=SEQ_LEN * 7,
            **weatherformer_size_params
        )
        if pretrained_weatherformer is not None:
            self.weather_transformer.in_proj = copy.deepcopy(
                pretrained_weatherformer.in_proj
            )
            self.weather_transformer.transformer_encoder = copy.deepcopy(
                pretrained_weatherformer.transformer_encoder
            )
            self.weather_transformer.positional_encoding = copy.deepcopy(
                pretrained_weatherformer.positional_encoding
            )
            self.weather_transformer.input_scaler = copy.deepcopy(
                pretrained_weatherformer.input_scaler
            )
            # self.weather_transformer.temporal_pos_encoding = copy.deepcopy(
            #     pretrained_weatherformer.temporal_pos_encoding
            # )

        self.trend_transformer = TransformerModel(
            input_dim=hidden_dim + 4,
            output_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Fully connected layer to output the predicted flu cases
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        weather,
        mask,
        weather_index,
        coord,
        ili_past,
        tot_cases_past,
    ):
        batch_size, seq_len, n_features = weather.size()
        weather_feature_mask = torch.zeros(
            (self.weather_transformer.input_dim,),
            device=DEVICE,
            dtype=torch.bool,
        )

        weather = self.weather_transformer(
            weather,
            coord,
            weather_index,
            weather_feature_mask=weather_feature_mask,
            src_key_padding_mask=mask,
        )

        # Concatenate processed weather, last year's same week flu cases, and last week's flu cases
        combined_input = torch.cat(
            [
                weather,
                ili_past.unsqueeze(2),
                tot_cases_past.unsqueeze(2),
                coord.unsqueeze(1).expand(-1, weather.shape[1], -1),
            ],
            dim=2,
        )
        output = self.trend_transformer(combined_input)

        # Predict current week's flu cases
        predicted_flu_cases = self.fc(output)
        return predicted_flu_cases.squeeze()
