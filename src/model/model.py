import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
from typing import Optional


class WFPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device=DEVICE):
        assert (
            d_model % 4 == 0
        ), "d_model should be divisible by 4 for separate encoding"

        # we did seq_len + 1 for max len cause there's a summary vector
        super(WFPositionalEncoding, self).__init__()

        # Create a position array (time encoding)
        self.position_list = (
            torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        )

        # max is 10k**(-1/2), cause we have the bias
        self.div_term = torch.exp(
            torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model)
        ).to(device)

    def forward(self, token_embedding, coords, device=DEVICE):
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
        latitude, longitude = coords[:, :1], coords[:, 1:]
        # Normalize latitude and longitude
        lat_norm = (latitude / 180.0) * math.pi
        lon_norm = (longitude / 180.0) * math.pi

        # Create geo encoding
        custom_pe = torch.zeros(batch_size, seq_len, d_model, device=device)

        # geo pe
        custom_pe[:, :, 2::4] = torch.sin(lat_norm * self.div_term).unsqueeze(1)
        custom_pe[:, :, 3::4] = torch.cos(lon_norm * self.div_term).unsqueeze(1)

        time_frequency = (self.position_list * self.div_term).unsqueeze(0)
        # encode time in 4k and 4k + 1
        custom_pe[:, :, 0::4] = torch.sin(time_frequency)
        custom_pe[:, :, 1::4] = torch.cos(time_frequency)

        # Add positional encoding to input
        token_embedding += custom_pe
        return token_embedding


# class WFSelfAttention(nn.MultiheadAttention):
#     def __init__(self, embed_dim, num_heads, dropout=0.1, device=None, dtype=None):
#         super(WFSelfAttention, self).__init__(
#             embed_dim,
#             num_heads,
#             dropout=dropout,
#             device=device,
#             dtype=dtype,
#             batch_first=True,
#         )
#         factory_kwargs = {"device": device, "dtype": dtype}

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.batch_first = True
#         assert (
#             self.head_dim * num_heads == embed_dim
#         ), "embed_dim must be divisible by num_heads"

#         self.scaling = self.head_dim**-0.5

#         self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, **factory_kwargs)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

#         self.dropout = nn.Dropout(dropout)

#         self.granularity_embed = nn.Embedding(
#             num_embeddings=31, embedding_dim=embed_dim
#         )  # up to 30 days + 1 for padding

#         # Initialize weights
#         self._init_parameters()

#     def _init_parameters(self):
#         # Xavier uniform initialization for the linear layers
#         nn.init.xavier_uniform_(self.qkv_proj.weight)

#     def forward(
#         self,
#         x,
#         temporal_granularity,
#         key_padding_mask=None,
#         need_weights=True,
#         attn_mask=None,
#         average_attn_weights=True,
#     ):
#         batch_size, seq_length, embed_dim = x.size()

#         # self attention uses same x to create Q, K, V
#         qkv = self.qkv_proj(x)
#         qkv = (
#             qkv.unflatten(-1, (3, embed_dim))
#             .unsqueeze(0)
#             .transpose(0, -2)
#             .squeeze(-2)
#             .contiguous()
#         )
#         qkv = qkv.view(
#             3, seq_length, batch_size * self.num_heads, self.head_dim
#         ).transpose(1, 2)

#         q, k, v = qkv[0], qkv[1], qkv[2]

#         gran_embed = self.granularity_embed(temporal_granularity).view(
#             batch_size * self.num_heads, 1, self.head_dim
#         )

#         q += gran_embed
#         k += gran_embed

#         if key_padding_mask is not None:
#             key_padding_mask = F._canonical_mask(
#                 mask=key_padding_mask,
#                 mask_name="key_padding_mask",
#                 other_type=F._none_or_dtype(attn_mask),
#                 other_name="attn_mask",
#                 target_type=q.dtype,
#             )
#             print(key_padding_mask.shape)
#             print(q.shape)
#             print(k.transpose(-2, -1).shape)
#             attn_scores = (
#                 torch.baddbmm(key_padding_mask, q, k.transpose(-2, -1)) * self.scaling
#             )
#         else:
#             # Calculate scores
#             attn_scores = torch.bmm(q, k.transpose(-2, -1)) * self.scaling

#         # Apply attention mask if provided
#         # if attn_mask is not None:
#         #     attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

#         # # Apply key padding mask
#         # if key_padding_mask is not None:
#         #     attn_scores = attn_scores.view(
#         #         batch_size, self.num_heads, seq_length, seq_length
#         #     )
#         #     attn_scores = attn_scores.masked_fill(
#         #         key_padding_mask.unsqueeze(1).unsqueeze(2),
#         #         float("-inf"),
#         #     )
#         #     attn_scores = attn_scores.view(
#         #         batch_size * self.num_heads, seq_length, seq_length
#         #     )

#         # Softmax to get the weights
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)

#         # Multiply weights by values
#         attn_output = torch.bmm(attn_weights, v)

#         # Concatenate heads and put through final linear layer
#         attn_output = (
#             attn_output.transpose(0, 1)
#             .contiguous()
#             .view(batch_size * seq_length, embed_dim)
#         )
#         attn_output = self.out_proj(attn_output).view(batch_size, seq_length, embed_dim)

#         if need_weights:
#             # Optionally return attention weights
#             attn_weights = attn_weights.view(
#                 batch_size, self.num_heads, seq_length, seq_length
#             )
#             if average_attn_weights:
#                 attn_weights = attn_weights.mean(dim=1)

#             return attn_output, attn_weights
#         else:
#             return attn_output


class Weatherformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=16,
        num_layers=6,
        hidden_dim_factor=16,
        max_len=CONTEXT_LENGTH,
    ):
        super(Weatherformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.input_scaler = nn.Embedding(
            num_embeddings=MAX_GRANULARITY_DAYS, embedding_dim=input_dim, padding_idx=0
        )
        torch.nn.init.constant_(self.input_scaler.weight.data, 1.0)

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = WFPositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            device=DEVICE,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        weather,
        coords,
        temporal_index,
        weather_feature_mask=None,  # n_features,
        src_key_padding_mask=None,  # batch_size x seq_len
    ):

        batch_size, seq_len, n_features = weather.shape

        # temporal index is index in time and temporal granularity ()
        temporal_granularity = temporal_index[:, 1].int()
        temp_embedding = self.input_scaler(temporal_granularity)

        # mask certain features in the input weather
        if weather_feature_mask is not None:
            # scalers for for masked dimensions = true becomes zero
            temp_embedding = (~weather_feature_mask).unsqueeze(0) * temp_embedding

        # mask the masked dimensions and scale the rest
        weather = weather * temp_embedding.view(batch_size, 1, n_features)
        weather[:, :, weather_feature_mask] = 0

        weather = self.in_proj(weather)
        weather = self.positional_encoding(weather, coords)
        weather = self.transformer_encoder(
            weather, src_key_padding_mask=src_key_padding_mask
        )
        weather = self.out_proj(weather)

        return weather
