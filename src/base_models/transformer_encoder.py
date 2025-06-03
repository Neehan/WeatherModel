from src.base_models.vanilla_pos_encoding import VanillaPositionalEncoding
from src.utils.constants import DEVICE

import torch
import torch.nn as nn
import copy
import math


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=8,
        num_layers=3,
        hidden_dim_factor=8,
    ):
        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = VanillaPositionalEncoding(
            hidden_dim, max_len=5000, device=DEVICE
        )
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.attn_layer = nn.Linear(hidden_dim, 1)  # Learnable attention layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_tensor, coord, mask=None, return_sequence=False):
        embedded_tensor = self.embedding(input_tensor)
        encoded_tensor = self.positional_encoding(embedded_tensor, coord)
        encoded_tensor = self.transformer_encoder(
            encoded_tensor, src_key_padding_mask=mask
        )
        # Compute attention weights
        attn_weights = self.attn_layer(encoded_tensor)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_summary = torch.sum(encoded_tensor * attn_weights, dim=1)
        weighted_summary = self.fc(weighted_summary)

        # Check if the full sequence should be returned
        if return_sequence:
            # Multiply entire encoded tensor with self.fc
            encoded_tensor = self.fc(encoded_tensor)
            return encoded_tensor, weighted_summary
        else:
            return weighted_summary
