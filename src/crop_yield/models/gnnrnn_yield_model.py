import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from typing import Optional

from src.base_models.base_model import BaseModel
from src.utils.utils import normalize_year_interval_coords


class EncoderModule(nn.Module):
    def __init__(self):
        super(EncoderModule, self).__init__()
        # Updated to match dataloader format
        self.time_intervals = 52  # 52 weeks
        self.soil_depths = 6
        self.num_weather_vars = 6
        self.num_soil_vars = 11

        # Weather CNN (weekly data) - input: (batch, 6, 52*n_years)
        self.weather_conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, 3, 1),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(128, 256, 3, 1),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(256, 512, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(
                1
            ),  # Adaptive pooling to handle variable sequence lengths
        )

        # Soil CNN - input: (batch, 11, 6*n_years)
        self.soil_conv = nn.Sequential(
            nn.Conv1d(in_channels=11, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Adaptive pooling
        )

        # Feature projections for CNN outputs
        self.weather_proj = nn.Linear(512, 64)
        self.soil_proj = nn.Linear(64, 32)

        # Total feature size: 64 (weather) + 32 (soil) + 2 (coords) + 1 (year) + 1 (past_yield) = 100
        self.total_features = 100

        # Project final concatenated features to z_dim
        self.feature_proj = nn.Linear(100, 128)

    def forward(self, weather, soil, coords, year, interval, past_yield):
        """
        Args:
            weather: (batch, n_years, 6, 52)
            soil: (batch, n_years, 11, 6)
            coords: (batch, 2) - UNNORMALIZED
            year: (batch, seq_len) - UNNORMALIZED years
            interval: (batch, 1) - UNNORMALIZED interval in days
            past_yield: (batch, 1) - past yield for this timestep
        """
        batch_size = weather.shape[0]
        n_years = weather.shape[1]

        # Normalize coordinates, year, and interval (same as other models)
        year_norm, interval_norm, coords_norm = normalize_year_interval_coords(
            year, interval, coords
        )

        # Take mean year for this timestep (like CNN-RNN model)
        year_mean = year_norm.mean(dim=1, keepdim=True)  # (batch, 1)

        # Reshape weather: (batch, n_years, 6, 52) -> (batch, 6, n_years*52)
        weather = weather.transpose(1, 2).contiguous()  # (batch, 6, n_years, 52)
        weather = weather.view(batch_size, 6, -1)  # (batch, 6, n_years*52)

        # Reshape soil: (batch, n_years, 11, 6) -> (batch, 11, n_years*6)
        soil = soil.transpose(1, 2).contiguous()  # (batch, 11, n_years, 6)
        soil = soil.view(batch_size, 11, -1)  # (batch, 11, n_years*6)

        # Process through CNNs
        weather_feat = self.weather_conv(weather).squeeze(-1)  # (batch, 512)
        soil_feat = self.soil_conv(soil).squeeze(-1)  # (batch, 64)

        # Project features
        weather_feat = self.weather_proj(weather_feat)  # (batch, 64)
        soil_feat = self.soil_proj(soil_feat)  # (batch, 32)

        # Concatenate everything: weather CNN + soil CNN + coords + year + past_yield (like CNN-RNN)
        features = torch.cat(
            [
                weather_feat,  # (batch, 64)
                soil_feat,  # (batch, 32)
                coords_norm,  # (batch, 2) - normalized coords, no projection
                year_mean,  # (batch, 1) - mean year, no projection
                past_yield,  # (batch, 1) - past yield
            ],
            dim=1,
        )  # (batch, 64+32+2+1+1 = 100)

        # Project to match z_dim
        features = self.feature_proj(features)  # (batch, 128)
        return features


class GNNRNNYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int = 6,  # Not used but kept for compatibility
        n_past_years: int = 5,
        **kwargs,
    ):
        super().__init__(name)
        self.device = device
        self.n_past_years = n_past_years

        # Model hyperparameters
        self.z_dim = 128
        self.n_layers = 3
        self.aggregator_type = "mean"
        self.dropout_prob = 0.1

        # Encoder module for processing weather, soil, coords
        self.encoder = EncoderModule()

        # GraphSAGE layers - input is encoder features (128, past yield already included)
        self.layers = nn.ModuleList()
        # First layer: input features (128, past yield included in encoder) -> hidden_dim
        self.layers.append(dglnn.SAGEConv(self.z_dim, self.z_dim, self.aggregator_type))

        # Middle layers: hidden_dim -> hidden_dim
        for i in range(1, self.n_layers - 1):
            self.layers.append(
                dglnn.SAGEConv(self.z_dim, self.z_dim, self.aggregator_type)
            )

        # Last layer: hidden_dim -> hidden_dim
        if self.n_layers > 1:
            self.layers.append(
                dglnn.SAGEConv(self.z_dim, self.z_dim, self.aggregator_type)
            )

        self.dropout = nn.Dropout(self.dropout_prob)

        # LSTM for temporal processing across years
        self.lstm = nn.LSTM(
            input_size=self.z_dim,
            hidden_size=self.z_dim,
            num_layers=1,
            batch_first=True,
        )

        # Final regression layers - takes LSTM output (past yields already incorporated at each timestep)
        self.regressor = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.z_dim // 2, 1),
        )

    def forward(
        self,
        weather,
        soil,
        coords,
        past_yields,
        blocks=None,  # DGL blocks for sampling
    ):
        """
        Forward pass accepting GNN data format directly

        Args:
            weather: (batch, n_years, 6, 52) - direct from GNN dataloader
            soil: (batch, n_years, 11, 6) - direct from GNN dataloader
            coords: (batch, 2) - coordinates
            past_yields: (batch, n_past_years + 1) - past yields
            blocks: DGL blocks for graph sampling
        """
        batch_size = weather.shape[0]
        n_years = weather.shape[1]

        # Process each year: Encoder (with past yield) → GraphSAGE
        hs = []
        for i in range(n_years):
            # Process year i
            year_weather = weather[:, i : i + 1]  # (batch, 1, 6, 52)
            year_soil = soil[:, i : i + 1]  # (batch, 1, 11, 6)

            # Get past yield for this timestep
            past_yield_i = past_yields[:, i : i + 1]  # (batch, 1)

            # Dummy inputs for encoder compatibility
            year_expanded = torch.zeros(batch_size, 52).to(weather.device)
            interval = torch.zeros(batch_size, 1).to(weather.device)

            # Encode features using encoder module
            h = self.encoder(
                year_weather,
                year_soil,
                coords,
                year_expanded,
                interval,
                past_yield_i,
            )  # (batch, 128)

            # Apply GraphSAGE layers (no need to add past yield again)
            if blocks is not None:
                for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                    if l < len(blocks):
                        h_dst = h[: block.number_of_dst_nodes()]
                        h = layer(block, (h, h_dst))
                    else:
                        # For layers without corresponding blocks, use full graph
                        h = layer(self.g, h)

                    if l != len(self.layers) - 1:
                        h = F.relu(h)
                        h = self.dropout(h)
            # If no blocks, h remains as encoded features

            hs.append(h)  # Store processed features for this year

        # Stack all years and process through LSTM
        hs = torch.stack(hs, dim=1)  # (batch, n_years, z_dim)

        # LSTM for temporal modeling across years
        lstm_out, _ = self.lstm(hs)  # (batch, n_years, z_dim)

        # Use final timestep output for prediction (past yields already incorporated at each timestep)
        lstm_final = lstm_out[:, -1]  # (batch, z_dim)

        # Final prediction (past yields already processed through encoder at each timestep)
        pred = self.regressor(lstm_final)  # (batch, 1)

        return pred
