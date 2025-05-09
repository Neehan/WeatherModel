from ..model.bert_model import WeatherBERT
from .constants import *
from .model import PositionalEncoding, TransformerModel
import torch
import torch.nn as nn
import copy


class BERTFluPredictor(nn.Module):
    def __init__(
        self,
        pretrained_weatherformer: WeatherBERT,
        weatherformer_size_params,
        n_predict_weeks=5,
        hidden_dim=32,
        num_layers=3,
    ):
        super().__init__()

        self.weather_transformer = WeatherBERT(
            len(WEATHER_PARAMS),
            hidden_dim,
            max_len=365,
            **weatherformer_size_params,
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
            self.weather_transformer.max_len = pretrained_weatherformer.max_len

        self.trend_transformer = TransformerModel(
            input_dim=hidden_dim + 2,  # flu features
            output_dim=n_predict_weeks,
            num_layers=num_layers,
        )

        # Fully connected layer to output the predicted flu cases
        # self.fc = nn.Linear(hidden_dim, n_predict_weeks)

    def forward(
        self,
        weather,
        mask,
        weather_index,
        coords,
        ili_past,
        tot_cases_past,
    ):
        batch_size, seq_len, n_features = weather.size()
        # weather_feature_mask = torch.zeros(
        #     (self.weather_transformer.input_dim,),
        #     device=DEVICE,
        #     dtype=torch.bool,
        # )
        weather_feature_mask = torch.ones(
            (self.weather_transformer.input_dim,),
            device=DEVICE,
            dtype=torch.bool,
        )
        weather_indices = torch.tensor(
            [
                0,
                # 4,
                # 6,
                # 7,
                # 8,
                # 24, 25
            ],
            device=DEVICE,
            dtype=torch.int,
        )
        masked_weather = torch.zeros(weather.shape, device=DEVICE)
        masked_weather[:, :, weather_indices] = weather[:, :, weather_indices]
        weather_feature_mask = torch.ones(
            (batch_size, seq_len, self.weather_transformer.input_dim),
            dtype=torch.bool,
            device=DEVICE,
        )
        weather_feature_mask[:, :, weather_indices] = False

        weather = self.weather_transformer(
            masked_weather,
            coords,
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
                # coords.unsqueeze(1).expand(-1, weather.shape[1], -1),
            ],
            dim=2,
        )
        output = self.trend_transformer(combined_input, mask=mask)
        output[:, :1] += ili_past[:, -1:]

        return output
