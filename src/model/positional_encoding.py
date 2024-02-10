import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=52):
        super(PositionalEncoding, self).__init__()

        assert (
            d_model % 4 == 0
        ), "d_model should be divisible by 4 for separate encoding"

        # Info
        self.dropout = nn.Dropout(p=0.1)
        # Create a position array (time encoding)
        position_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # position 0 and 52 are identical weather has a yearly cyclic pattern
        position_list = position_list / max_len * 2 * math.pi

        self.div_term = torch.exp(
            torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model)
        )

        time_pe = torch.zeros(max_len, d_model)
        # encode time in 4k and 4k+1
        time_pe[:, 0::4] = torch.sin(position_list * self.div_term)
        time_pe[:, 1::4] = torch.cos(position_list * self.div_term)

        # save time_pe without gradients
        self.register_buffer("time_pe", time_pe)

    def forward(self, token_embedding, coords):
        """
        Forward method for adding positional encoding.

        Args:
        token_embedding: Tensor, shape [batch_size, seq_len, d_model]
        latitude: Tensor, shape [batch_size, 1]
        longitude: Tensor, shape [batch_size, 1]

        Returns:
        Tensor with positional encoding added, same shape as x.
        """
        batch_size, seq_len, d_model = token_embedding.shape

        latitude, longitude = coords[:, 0], coords[:, 1]

        # Normalize latitude and longitude
        lat_norm = (latitude / 180.0) * math.pi
        lon_norm = (longitude / 180.0) * math.pi

        # Create geo encoding
        geo_pe = torch.zeros(batch_size, seq_len, d_model)
        geo_pe[:, :, 2::4] = torch.sin(lat_norm).unsqueeze(1)  # Latitude encoding (sin)
        geo_pe[:, :, 3::4] = torch.cos(lon_norm).unsqueeze(
            1
        )  # Longitude encoding (cos)

        # Add positional encoding to input
        token_embedding = (
            token_embedding + self.time_pe[:seq_len, :].unsqueeze(0) + geo_pe
        )
        return self.dropout(token_embedding)
