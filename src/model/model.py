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


class WFSelfAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.1, device=None, dtype=None):
        super(WFSelfAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype,
            batch_first=True,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = True
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        self.dropout = nn.Dropout(dropout)

        self.granularity_embed = nn.Embedding(
            num_embeddings=31, embedding_dim=embed_dim
        )  # up to 30 days + 1 for padding

        # Initialize weights
        self._init_parameters()

    def _init_parameters(self):
        # Xavier uniform initialization for the linear layers
        nn.init.xavier_uniform_(self.qkv_proj.weight)

    def forward(
        self,
        x,
        temporal_granularity,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        batch_size, seq_length, embed_dim = x.size()

        # self attention uses same x to create Q, K, V
        qkv = self.qkv_proj(x)
        qkv = (
            qkv.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        qkv = qkv.view(
            3, seq_length, batch_size * self.num_heads, self.head_dim
        ).transpose(1, 2)

        q, k, v = qkv[0], qkv[1], qkv[2]

        gran_embed = self.granularity_embed(temporal_granularity).view(
            batch_size * self.num_heads, 1, self.head_dim
        )

        q += gran_embed
        k += gran_embed

        # Calculate scores
        attn_scores = torch.bmm(q, k.transpose(-2, -1)) * self.scaling
        # each head

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.view(
                batch_size, self.num_heads, seq_length, seq_length
            )
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_scores = attn_scores.view(
                batch_size * self.num_heads, seq_length, seq_length
            )

        # Softmax to get the weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiply weights by values
        attn_output = torch.bmm(attn_weights, v)

        # Concatenate heads and put through final linear layer
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(batch_size * seq_length, embed_dim)
        )
        attn_output = self.out_proj(attn_output).view(batch_size, seq_length, embed_dim)

        if need_weights:
            # Optionally return attention weights
            attn_weights = attn_weights.view(
                batch_size, self.num_heads, seq_length, seq_length
            )
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)

            return attn_output, attn_weights
        else:
            return attn_output


class WFEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(WFEncoderLayer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
            batch_first=True,
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch_first = True

        # Using custom WFSelfAttention
        self.self_attn = WFSelfAttention(
            d_model, nhead, dropout=dropout, **factory_kwargs
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        temporal_granularity: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Forward pass through custom WFSelfAttention
        src2 = self.self_attn(
            src,
            temporal_granularity,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            need_weights=False,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Forward pass through Feedforward Network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class WFTransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ):
        super().__init__(
            encoder_layer,
            num_layers,
            norm,
            enable_nested_tensor,
            mask_check,
        )

    def forward(
        self,
        src: torch.Tensor,
        temporal_granularity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        # Assume each layer can handle `additional_input` as the second positional argument.
        for mod in self.layers:
            output = mod(
                output,
                temporal_granularity,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class Weatherformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_heads=8, num_layers=3, hidden_dim_factor=8
    ):
        super(Weatherformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = CONTEXT_LENGTH
        # allow each weather feature to be scaled based on input mask
        self.input_scaler = nn.Parameter(torch.tensor([1.0] * input_dim, device=DEVICE))
        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = WFPositionalEncoding(
            hidden_dim, max_len=CONTEXT_LENGTH
        )
        encoder_layer = WFEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            device=DEVICE,
        )
        self.transformer_encoder = WFTransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        weather,
        coords,
        temporal_index,
        weather_feature_mask=None,
        src_key_padding_mask=None,
    ):

        # temporal index is index in time and temporal granularity ()
        temporal_granularity = temporal_index[:, 1:].unsqueeze(2).int()

        # mask certain features in the input weather
        if weather_feature_mask is not None:
            # scaler for masked dimensions = true becomes zero
            input_mask = (~weather_feature_mask) * self.input_scaler
            # mask the masked dimensions and scale the rest
            weather *= input_mask.view(1, 1, -1)

        weather = self.embedding(weather)
        weather = self.positional_encoding(weather, coords)
        weather = self.transformer_encoder(
            weather, temporal_granularity, src_key_padding_mask=src_key_padding_mask
        )
        weather = self.fc(weather)
        return weather
