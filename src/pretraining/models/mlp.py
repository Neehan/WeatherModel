import torch
import torch.nn as nn
from typing import Optional
from src.base_models.base_model import BaseModel
from src.utils.constants import MAX_CONTEXT_LENGTH

"""
Simple MLP model for baseline comparison.
Input: all 31 weather features
Output: 6 specific masked features [7, 8, 11, 1, 2, 29]
Architecture: 31 -> 128 -> GELU -> 128 -> GELU -> 6
"""


class MLP(BaseModel):
    def __init__(
        self,
        weather_dim,
        device,
        hidden_dim=128,
        max_len=MAX_CONTEXT_LENGTH,
    ):
        super(MLP, self).__init__("mlp")

        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Fixed masked features
        self.masked_features = [7, 8, 11, 1, 2, 29]
        self.output_dim = len(self.masked_features)

        # Simple MLP: weather_dim -> hidden -> hidden -> output_dim
        self.mlp = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

        self.to(device)

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
        weather: batch_size x seq_len x 31 (all features)
        Returns: batch_size x seq_len x 6 (only masked features)
        """
        batch_size, seq_len, n_features = weather.shape

        # Flatten to apply MLP to each timestep
        weather_flat = weather.reshape(-1, n_features)

        # Apply MLP - outputs only the 6 masked features
        output_flat = self.mlp(weather_flat)

        # Reshape back
        output = output_flat.reshape(batch_size, seq_len, self.output_dim)

        return output
