import math

import torch
import torch.nn as nn


class SpatiotemporalPositionalEncoding(nn.Module):
    # Type hints for registered buffers
    position_list: torch.Tensor
    div_term: torch.Tensor

    def __init__(self, d_model, max_len, device):
        assert (
            d_model % 4 == 0
        ), "d_model should be divisible by 4 for separate encoding"

        # we did seq_len + 1 for max len cause there's a summary vector
        super(SpatiotemporalPositionalEncoding, self).__init__()

        # Create a position array (time encoding) - create on CPU, register_buffer handles device
        position_list = torch.arange(0, max_len, dtype=torch.float).reshape(
            1, max_len, 1
        )

        # Register as buffer so it moves with the model to different devices
        self.register_buffer("position_list", position_list)

        # max is 10k**(-1/2), cause we have the bias - create on CPU, register_buffer handles device
        div_term = torch.exp(
            torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model)
        ).reshape(1, 1, -1)

        # Register as buffer so it moves with the model to different devices
        self.register_buffer("div_term", div_term)

    def forward(self, token_embedding, coords):
        """
        Forward method for adding positional encoding.

        Args:
        token_embedding: Tensor, shape [batch_size, seq_len, d_model]
        coords: Tensor, shape [batch_size, seq_len, 2] normalized to [-1, 1]
        coords must be normalized to [-1, 1]

        Returns:
        Tensor with positional encoding added, same shape as x.
        """
        batch_size, seq_len, d_model = token_embedding.shape
        latitude, longitude = coords[:, :, :1], coords[:, :, 1:]
        # Create geo encoding - use token_embedding.device to ensure same device
        custom_pe = torch.zeros(
            batch_size, seq_len, d_model, device=token_embedding.device
        )

        # geo pe
        custom_pe[:, :, 2::4] = torch.sin(latitude * self.div_term)
        custom_pe[:, :, 3::4] = torch.cos(longitude * self.div_term)

        time_frequency = self.position_list[:, :seq_len, :] * self.div_term
        # encode time in 4k and 4k + 1
        custom_pe[:, :, 0::4] = torch.sin(time_frequency)
        custom_pe[:, :, 1::4] = torch.cos(time_frequency)

        # Add positional encoding to input
        token_embedding += custom_pe
        return token_embedding
