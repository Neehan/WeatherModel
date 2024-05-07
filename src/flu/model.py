from ..model.model import Weatherformer
from .constants import *

import torch
import torch.nn as nn
import copy


class FluPredictor(nn.Module):
    def __init__(
        self,
        pretrained_weatherformer: Weatherformer = None,
        input_dim=128,
        hidden_dim=64,
        num_layers=1,
    ):
        super().__init__()

        self.weather_transformer = Weatherformer(WEATHER_PARAMS, 48, max_len=SEQ_LEN)
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
            self.weather_transformer.max_len = pretrained_weatherformer.max_len

        # LSTM for predicting flu cases
        self.lstm = nn.LSTM(
            input_size=input_dim
            + 2,  # weather features + past week's flu cases + last year same week
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
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
            (
                batch_size,
                seq_len,
                self.weather_transformer.input_dim,
            ),
            device=DEVICE,
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
                ili_past,
                tot_cases_past,
            ],
            dim=-1,
        )

        # LSTM to predict the current week's flu cases
        lstm_output, _ = self.lstm(combined_input)
        lstm_output = lstm_output[:, -1, :]  # Get the last time step output

        # Predict current week's flu cases
        predicted_flu_cases = self.fc(lstm_output)
        return predicted_flu_cases
