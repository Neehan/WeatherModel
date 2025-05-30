from src.base_models.transformer_encoder import TransformerEncoder
from src.utils.constants import *
from src.base_models.soil_cnn import SoilCNN

import torch
import torch.nn as nn
import copy
import math


class CNNYieldPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.soil_cnn = SoilCNN()

        self.weather_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=9, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=16, out_channels=20, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            # Flattening the output to fit Linear Layer
            nn.Flatten(),  # 20 x 1,
            nn.ReLU(),
        )

        self.weather_fc = nn.Sequential(nn.Linear(6 * 20, 60), nn.ReLU())
        self.soil_fc = nn.Sequential(
            nn.Linear(11 * 12, 40),
        )

        fc_dims = 60 + 40 + 14 + 1 + 1
        self.trend_transformer = TransformerEncoder(
            input_dim=fc_dims,
            output_dim=32,
            num_layers=3,
        )
        self.fc1 = nn.Linear(in_features=32, out_features=1)

    def forward(self, weather, soil, practices, year, coord, y_past, mask):

        batch_size, n_years, n_features, n_weeks = weather.size()
        weather = weather.reshape(batch_size * n_years * n_features, 1, n_weeks)
        weather = self.weather_cnn(weather)
        weather = weather.view(batch_size * n_years, -1)
        weather = self.weather_fc(weather)
        weather = weather.view(batch_size, n_years, -1)

        soil = soil.reshape(batch_size * n_years * soil.shape[2], 1, -1)
        soil_out = self.soil_cnn(soil)

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
