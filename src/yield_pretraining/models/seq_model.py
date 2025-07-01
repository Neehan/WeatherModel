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
            input_dim=5,  # year, lat, lon, period between points, yield
            output_dim=1,  # predict yield values
            num_heads=4,
            num_layers=3,
            hidden_dim_factor=8,
        )

    def forward(self, year, coords, period, yields, yield_mask, padding_mask):
        """
        Input shapes:
        year: [batch_size, seq_len]
        coords: [batch_size, seq_len, 2]
        period: [batch_size, seq_len]
        yields: [batch_size, seq_len] - contains known and unknown yields
        yield_mask: [batch_size, seq_len] - True for known yields, False for masked
        padding_mask: [batch_size, seq_len] - True for padding positions
        """
        # normalize
        year, period, coords = normalize_year_interval_coords(year, period, coords)

        # Unsqueeze to add feature dimension
        year = year.unsqueeze(2)  # [batch_size, seq_len, 1]
        period = period.unsqueeze(2)  # [batch_size, seq_len, 1]
        yields = yields.unsqueeze(2)  # [batch_size, seq_len, 1]
        # coords is already [batch_size, seq_len, 2]

        # Mask unknown yields to zero for input
        masked_yields = yields * yield_mask.unsqueeze(2).float()

        # Concatenate all features: [batch_size, seq_len, 5] (year + coords + period + yield)
        x = torch.cat([year, coords, period, masked_yields], dim=2)

        # Pass through transformer encoder with return_sequence=True to get predictions for all positions
        sequence_output, _ = self.encoder(x, mask=padding_mask, return_sequence=True)

        return sequence_output.squeeze(-1)  # [batch_size, seq_len]

    def load_pretrained(self, pretrained_model: "SeqModel"):
        """Load weights from a pretrained SeqModel by deep copying each layer."""
        self.logger.info("🔄 Loading yield sequence model from pretrained model")
        self.encoder = copy.deepcopy(pretrained_model.encoder)
