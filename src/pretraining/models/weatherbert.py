import copy
from typing import Optional

import torch
import torch.nn as nn

from src.base_models.base_model import BaseModel
from src.base_models.vanilla_pos_encoding import VanillaPositionalEncoding
from src.utils.constants import MAX_CONTEXT_LENGTH
from src.utils.utils import normalize_year_interval_coords


class WeatherBERT(BaseModel):
    def __init__(
        self,
        weather_dim,
        output_dim,
        device,
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
    ):
        super(WeatherBERT, self).__init__("weatherbert")

        self.weather_dim = weather_dim
        self.input_dim = weather_dim + 1 + 2  # weather (normalized) + (year-1970)/100
        self.output_dim = output_dim
        self.max_len = max_len

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.in_proj = nn.Linear(self.input_dim, hidden_dim)

        # self.spatiotemporal_embedding = nn.Sequential(
        #     nn.Linear(2, self.input_dim // 2),  # 2 for coords
        #     nn.GELU(),
        #     nn.Linear(self.input_dim // 2, self.input_dim),
        # )

        self.positional_encoding = VanillaPositionalEncoding(
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

    def load_pretrained(
        self, pretrained_model: "WeatherBERT", load_out_proj: bool = True
    ):
        """Load weights from a pretrained WeatherBERT model by deep copying each layer."""

        if self.input_dim != pretrained_model.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim} but received {pretrained_model.input_dim}"
            )
        if self.max_len != pretrained_model.max_len:
            raise ValueError(
                f"expected max length {self.max_len} but received {pretrained_model.max_len}"
            )

        self.in_proj = copy.deepcopy(pretrained_model.in_proj)
        self.positional_encoding = copy.deepcopy(pretrained_model.positional_encoding)
        self.transformer_encoder = copy.deepcopy(pretrained_model.transformer_encoder)
        # self.spatiotemporal_embedding = copy.deepcopy(
        #     pretrained_model.spatiotemporal_embedding
        # )
        if load_out_proj:
            self.logger.info("🔄 Loading out_proj from pretrained model")
            self.out_proj = copy.deepcopy(pretrained_model.out_proj)
        else:
            self.logger.info("⚠️ Not loading out_proj from pretrained model")

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len
    ) -> torch.Tensor:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2 (lat, lon) UNNORMALIZED
        year: batch_size x seq_len (UNNORMALIZED, time-varying years)
        interval: batch_size x 1 (UNNORMALIZED in days)
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len
        """
        batch_size, seq_len, n_features = weather.shape
        # normalize year, interval, and coords
        year, interval, coords = normalize_year_interval_coords(year, interval, coords)

        # Year is [batch_size, seq_len], add feature dimension to make it [batch_size, seq_len, 1]
        year = year.unsqueeze(2)
        # Expand coords to match sequence length if needed
        coords = coords.unsqueeze(1).expand(batch_size, seq_len, 2)

        # mask weather for the masked dimensions
        weather = weather * (~weather_feature_mask)

        input_tensor = torch.cat([weather, year, coords], dim=2)
        input_tensor = self.in_proj(input_tensor)
        input_tensor = self.positional_encoding(input_tensor)
        input_tensor = self.transformer_encoder(
            input_tensor, src_key_padding_mask=src_key_padding_mask
        )
        output = self.out_proj(input_tensor)

        return output
