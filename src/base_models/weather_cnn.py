import torch
import torch.nn as nn
import copy
from typing import Optional

from src.base_models.base_model import BaseModel
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE, TOTAL_WEATHER_VARS
from src.utils.utils import normalize_year_interval_coords


class WeatherCNN(BaseModel):
    def __init__(
        self,
        weather_dim,
        output_dim,
        max_len,
        device=DEVICE,
    ):
        super(WeatherCNN, self).__init__("weathercnn")

        self.weather_dim = weather_dim
        self.input_dim = (
            weather_dim + 2 + 1 + 1
        )  # weather_dim + coords + year + interval
        self.output_dim = output_dim
        self.max_len = max_len
        self.device = device

        # CNN layers for processing weather sequences
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
            nn.Flatten(),  # 20 x calculated_size
            nn.ReLU(),
        )

        # Calculate the final sequence length after all CNN operations
        # Use a dummy input to calculate CNN output features per input dimension
        dummy_input = torch.zeros(self.input_dim, 1, max_len)
        with torch.no_grad():
            dummy_output = self.weather_cnn(dummy_input)
        self.final_fc_input_size = dummy_output.shape[1] * self.input_dim
        # Final combination layer - input_dim * cnn_features_per_input
        self.final_fc = nn.Linear(self.final_fc_input_size, output_dim)

    def load_pretrained(self, pretrained_model: "WeatherCNN"):
        """Load weights from a pretrained WeatherCNN model by deep copying each layer."""

        if self.weather_dim != pretrained_model.weather_dim:
            raise ValueError(
                f"expected weather dimension {self.weather_dim} but received {pretrained_model.weather_dim}"
            )
        if self.output_dim != pretrained_model.output_dim:
            raise ValueError(
                f"expected output dimension {self.output_dim} but received {pretrained_model.output_dim}"
            )
        if self.max_len != pretrained_model.max_len:
            raise ValueError(
                f"expected max length {self.max_len} but received {pretrained_model.max_len}"
            )
        if self.final_fc_input_size != pretrained_model.final_fc_input_size:
            raise ValueError(
                f"expected final FC input size {self.final_fc_input_size} but received {pretrained_model.final_fc_input_size}"
            )

        self.weather_cnn = copy.deepcopy(pretrained_model.weather_cnn)
        self.final_fc = copy.deepcopy(pretrained_model.final_fc)

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
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
        if n_features != self.weather_dim:
            raise ValueError(
                f"expected {self.weather_dim} weather features but received {n_features} features"
            )

        # Normalize year, interval, and coords
        year, interval, coords = normalize_year_interval_coords(year, interval, coords)

        # Apply masking if provided
        if (
            weather_feature_mask is not None
            and weather_feature_mask.shape == weather.shape
        ):
            weather = weather * (~weather_feature_mask)

        # Add year as feature dimension to match BERT approach
        year = year.unsqueeze(2)  # batch_size x seq_len x 1

        # Expand interval and coords to match sequence length
        interval = interval.unsqueeze(1).expand(
            batch_size, seq_len, 1
        )  # batch_size x seq_len x 1
        coords = coords.unsqueeze(1).expand(
            batch_size, seq_len, 2
        )  # batch_size x seq_len x 2

        # Concatenate all features: weather + coords + year + interval
        input_tensor = torch.cat(
            [weather, coords, year, interval], dim=2
        )  # batch_size x seq_len x input_dim

        # Process each feature dimension through CNN while preserving sequence
        # Transpose to (batch_size * input_dim, 1, seq_len) for conv1d processing
        input_tensor = input_tensor.transpose(1, 2)  # batch_size x input_dim x seq_len
        batch_size, input_dim, seq_len = input_tensor.shape

        # Reshape to process each feature separately: (batch_size * input_dim, 1, seq_len)
        input_tensor = input_tensor.reshape(batch_size * input_dim, 1, seq_len)

        # Pass all features through CNN in parallel
        cnn_output = self.weather_cnn(
            input_tensor
        )  # (batch_size * input_dim, features)

        # Reshape back to (batch_size, final_fc_input_size)
        cnn_output = cnn_output.reshape(batch_size, self.final_fc_input_size)

        # Pass through final FC layer
        final_output = self.final_fc(cnn_output)

        return final_output
