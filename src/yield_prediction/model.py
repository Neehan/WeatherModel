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
            dim_model % 4 == 0
        ), "dim_model should be divisible by 4 for separate encoding"
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(p=0.1)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model).to(DEVICE)
        positions_list = (
            torch.arange(0, max_len, dtype=torch.float).view(-1, 1).to(DEVICE)
        )  # 0, 1, 2, 3, 4, 5
        self.div_term = torch.exp(
            torch.arange(0, dim_model, 4).float() * (-math.log(10000.0)) / dim_model
        ).to(
            DEVICE
        )  # 10000^(2i/dim_model)

        # PE(pos, 4i) = sin(pos/10000^(4i/dim_model))
        pos_encoding[:, 0::4] = torch.sin(positions_list * self.div_term)
        # PE(pos, 4i + 1) = cos(pos/1000^(4i/dim_model))
        pos_encoding[:, 1::4] = torch.cos(positions_list * self.div_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(
        self, token_embedding: torch.tensor, coords: torch.tensor
    ) -> torch.tensor:

        batch_size, seq_len, d_model = token_embedding.shape
        latitude, longitude = coords[:, :1], coords[:, 1:]
        # Normalize latitude and longitude
        lat_norm = (latitude / 180.0) * math.pi
        lon_norm = (longitude / 180.0) * math.pi

        # Create geo encoding
        geo_pe = torch.zeros(batch_size, seq_len, d_model, device=DEVICE)

        geo_pe[:, :, 2::4] = torch.sin(lat_norm * self.div_term).unsqueeze(1)
        geo_pe[:, :, 3::4] = torch.cos(lon_norm * self.div_term).unsqueeze(1)

        # Add positional encoding to input
        token_embedding = (
            token_embedding + self.pos_encoding[:seq_len, :].unsqueeze(0) + geo_pe
        )
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

    def forward(self, input_tensor, coord, mask=None, return_sequence=False):
        embedded_tensor = self.embedding(input_tensor)
        encoded_tensor = self.positional_encoding(embedded_tensor, coord)
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


class YieldPredictor(nn.Module):
    def __init__(self, pretrained_weatherformer: Weatherformer = None):
        super().__init__()
        self.soil_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),  #
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=8, out_channels=12, kernel_size=2, stride=1, padding=1
            ),
            # Flattening the output to fit Linear Layer
            nn.Flatten(),  # 24 x 1
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        self.soil_fc = nn.Sequential(
            nn.Linear(11 * 12, 40),
        )

        self.weather_transformer = Weatherformer(31, 48, max_len=SEQ_LEN)
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
            # self.weather_transformer.input_scaler = copy.deepcopy(
            #     pretrained_weatherformer.input_scaler
            # )
            self.weather_transformer.temporal_pos_encoding = copy.deepcopy(
                pretrained_weatherformer.temporal_pos_encoding
            )

            self.weather_transformer.max_len = pretrained_weatherformer.max_len

        self.weather_fc = nn.Sequential(
            nn.Linear(48 * SEQ_LEN, 120),
            # nn.ReLu()
        )

        fc_dims = 120 + 40 + 14 + 1 + 1
        self.trend_transformer = TransformerModel(
            input_dim=fc_dims,
            output_dim=32,
            num_layers=3,
        )
        self.fc1 = nn.Linear(in_features=32, out_features=1)

    def forward(self, weather, soil, practices, year, coord, y_past, mask):

        batch_size, n_years, n_features, seq_len = weather.size()
        weather = weather.view(batch_size * n_years, -1, n_features)

        coord = coord.view(batch_size * n_years, 2)
        year = year.view(batch_size * n_years, 1)
        # [7, 8, 11, 1, 2, 29] are the closest weather feature ids according to pretraining
        weather_indices = torch.tensor([7, 8, 11, 1, 2, 29])
        padded_weather = torch.zeros(
            (
                batch_size * n_years,
                seq_len,
                self.weather_transformer.input_dim,
            ),
            device=DEVICE,
        )
        padded_weather[:, :, weather_indices] = weather
        # create feature mask
        weather_feature_mask = torch.ones(
            self.weather_transformer.input_dim,
            dtype=torch.bool,
            device=DEVICE,
        )
        weather_feature_mask[weather_indices] = False

        # create temporal index
        temporal_gran = torch.full((batch_size * n_years, 1), 7, device=DEVICE)
        temporal_index = torch.cat([year, temporal_gran], dim=1)

        weather = self.weather_transformer(
            padded_weather,
            coord,
            temporal_index,
            weather_feature_mask=weather_feature_mask,
        )

        weather = weather.view(batch_size * n_years, -1)
        weather = self.weather_fc(weather)
        weather = weather.view(batch_size, n_years, -1)

        soil = soil.reshape(batch_size * n_years * soil.shape[2], 1, -1)
        soil_out = self.soil_cnn(soil)
        soil_out = soil_out.view(batch_size * n_years, -1)
        soil_out = self.soil_fc(soil_out)
        soil_out = soil_out.view(batch_size, n_years, -1)

        combined = torch.cat(
            (
                weather,
                soil_out,
                practices,
                year.reshape(batch_size, n_years, 1),
                y_past.unsqueeze(2),
            ),
            dim=2,
        )
        combined = self.trend_transformer(
            combined, coord.view(batch_size, -1, 2)[:, -1, :], mask
        )
        out = self.fc1(combined)
        return out
