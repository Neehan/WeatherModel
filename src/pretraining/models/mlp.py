import torch
import torch.nn as nn
from typing import Optional
from src.base_models.base_model import BaseModel
from src.utils.constants import MAX_CONTEXT_LENGTH

"""
This class implements the MLP model for baseline comparison.

Simple MLP architecture: 31 -> 128 (GELU) -> 31
Uses standard MSE loss for training.
"""


class MLP(BaseModel):
    def __init__(
        self,
        weather_dim,
        output_dim,
        device,
        hidden_dim=256,
        max_len=MAX_CONTEXT_LENGTH,
    ):
        super(MLP, self).__init__("mlp")

        self.weather_dim = weather_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Simple MLP: weather_dim -> hidden_dim -> output_dim
        self.mlp = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Move to device
        self.to(device)

    def load_pretrained(self, pretrained_model: "MLP", load_out_proj: bool = True):
        """Load weights from a pretrained MLP model."""
        if self.weather_dim != pretrained_model.weather_dim:
            raise ValueError(
                f"expected weather dimension {self.weather_dim} but received {pretrained_model.weather_dim}"
            )
        if self.output_dim != pretrained_model.output_dim:
            raise ValueError(
                f"expected output dimension {self.output_dim} but received {pretrained_model.output_dim}"
            )

        if load_out_proj:
            self.logger.info("🔄 Loading MLP weights from pretrained model")
            self.mlp.load_state_dict(pretrained_model.mlp.state_dict())
        else:
            self.logger.info("⚠️ Not loading MLP weights from pretrained model")

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2 (lat, lon) - NOT USED in MLP
        year: batch_size x seq_len - NOT USED in MLP
        interval: batch_size x 1 - NOT USED in MLP
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len - NOT USED in MLP

        Returns:
        output: batch_size x seq_len x n_features
        """
        batch_size, seq_len, n_features = weather.shape

        # Apply masking to weather features
        weather = weather * (~weather_feature_mask)

        # Reshape to apply MLP to each timestep independently
        # [batch_size, seq_len, n_features] -> [batch_size * seq_len, n_features]
        weather_flat = weather.view(-1, n_features)

        # Apply MLP
        output_flat = self.mlp(weather_flat)

        # Reshape back to original dimensions
        # [batch_size * seq_len, n_features] -> [batch_size, seq_len, n_features]
        output = output_flat.view(batch_size, seq_len, n_features)

        return output
