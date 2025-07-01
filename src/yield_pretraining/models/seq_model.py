import copy
import torch
import torch.nn as nn
from src.base_models.base_model import BaseModel
from src.base_models.transformer_encoder import TransformerEncoder
from src.utils.utils import normalize_year_interval_coords


class SeqModel(BaseModel):
    def __init__(self, name: str, device: torch.device):
        super().__init__(name)
        self.encoder = TransformerEncoder(
            input_dim=5,  # year, lat, lon, period between points, past yield
            output_dim=1,  # hidden dimension for features
            num_heads=4,
            num_layers=3,
            hidden_dim_factor=8,
        )

    def forward(self, year, coords, period, past_yield):
        """
        Input shapes:
        year: [batch_size, seq_len]
        coords: [batch_size, seq_len, 2]
        period: [batch_size, seq_len]
        past_yield: [batch_size, seq_len]
        """
        # normalize
        year, period, coords = normalize_year_interval_coords(year, period, coords)

        # Unsqueeze to add feature dimension
        year = year.unsqueeze(2)  # [batch_size, seq_len, 1]
        period = period.unsqueeze(2)  # [batch_size, seq_len, 1]
        past_yield = past_yield.unsqueeze(2)  # [batch_size, seq_len, 1]
        # coords is already [batch_size, seq_len, 2]

        # Concatenate all features: [batch_size, seq_len, 5] (year + coords + period + past_yield)
        x = torch.cat([year, coords, period, past_yield], dim=2)

        # Pass through transformer encoder - pass concatenated features and coords
        x = self.encoder(x)

        # Add the predicted change in yield to the last past yield
        return past_yield[:, -1:, 0] + x

    def load_pretrained(self, pretrained_model: "SeqModel"):
        """Load weights from a pretrained SeqModel by deep copying each layer."""
        self.logger.info("🔄 Loading yield sequence model from pretrained model")
        self.encoder = copy.deepcopy(pretrained_model.encoder)
