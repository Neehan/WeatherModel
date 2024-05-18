from ..model.bert_model import WeatherBERT
from .constants import *
from .model import TransformerModel

import torch
import torch.nn as nn
import copy


class BERTYieldPredictor(nn.Module):
    def __init__(
        self, pretrained_weatherformer: WeatherBERT, weatherformer_size_params
    ):
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

        self.weather_transformer = WeatherBERT(
            31, 48, max_len=SEQ_LEN, **weatherformer_size_params
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

            self.weather_transformer.max_len = pretrained_weatherformer.max_len

        self.weather_fc = nn.Sequential(
            nn.Linear(48 * SEQ_LEN, 120),
            # nn.ReLu()
        )

        fc_dims = 120 + 40 + 14 + 1 + 1 + 2
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
            (batch_size * n_years, seq_len, self.weather_transformer.input_dim),
            dtype=torch.bool,
            device=DEVICE,
        )
        weather_feature_mask[:, :, weather_indices] = False

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
                coord.view(batch_size, n_years, 2),
            ),
            dim=2,
        )
        combined = self.trend_transformer(
            combined, coord.view(batch_size, -1, 2)[:, -1, :], mask
        )
        out = self.fc1(combined)
        return out
