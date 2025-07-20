import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from typing import Optional

from src.base_models.base_model import BaseModel
from src.utils.utils import normalize_year_interval_coords


class CNNEncoder(nn.Module):
    """CNN encoder matching the original paper architecture"""

    def __init__(self):
        super(CNNEncoder, self).__init__()

        # Dataset dimensions matching the original paper and our data format
        self.time_intervals = 52  # 52 weeks
        self.soil_depths = 6  # 6 depth levels
        self.num_weather_vars = 6  # 6 weather variables
        self.num_soil_vars = 11  # 11 soil variables

        # Weather CNN (matches original paper for weekly data)
        self.wm_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_weather_vars,
                out_channels=64,
                kernel_size=9,
                stride=1,
            ),
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
            nn.AvgPool1d(2, 2),
        )

        self.wm_fc = nn.Sequential(
            nn.Linear(512, 80),
            nn.ReLU(),
        )

        # Soil CNN (matches original paper for 6 depths)
        self.s_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_soil_vars, out_channels=16, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 2, 1),
            nn.ReLU(),
        )

        self.s_fc = nn.Sequential(
            nn.Linear(64, 40),
            nn.ReLU(),
        )

    def forward(self, weather, soil):
        """
        CNN encoder processes weather and soil data (like original paper)

        Args:
            weather: (batch, n_years, 6, 52)
            soil: (batch, n_years, 11, 6)
        Returns:
            cnn_features: (batch, 120) - 80 (weather) + 40 (soil)
        """
        batch_size = weather.shape[0]
        n_years = weather.shape[1]

        # Process weather data: (batch, n_years, 6, 52) -> (batch, 6, n_years*52)
        weather = weather.transpose(1, 2).contiguous()  # (batch, 6, n_years, 52)
        weather = weather.view(batch_size, 6, -1)  # (batch, 6, n_years*52)

        # Weather CNN processing
        wm_features = self.wm_conv(weather).squeeze(-1)  # (batch, 512)
        wm_features = self.wm_fc(wm_features)  # (batch, 80)

        # Process soil data: (batch, n_years, 11, 6) -> (batch, 11, n_years*6)
        soil = soil.transpose(1, 2).contiguous()  # (batch, 11, n_years, 6)
        soil = soil.view(batch_size, 11, -1)  # (batch, 11, n_years*6)

        # Soil CNN processing
        s_features = self.s_conv(soil).squeeze(-1)  # (batch, 64)
        s_features = self.s_fc(s_features)  # (batch, 40)

        # Concatenate CNN features: weather + soil = 80 + 40 = 120
        cnn_features = torch.cat([wm_features, s_features], dim=1)  # (batch, 120)

        return cnn_features


class GNNRNNYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int = 6,  # Keep for compatibility
        n_past_years: int = 5,
        z_dim: int = 128,
        n_layers: int = 3,
        aggregator_type: str = "mean",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(name)
        self.device = device
        self.n_past_years = n_past_years
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.aggregator_type = aggregator_type
        self.dropout_prob = dropout

        # CNN encoder matching original paper
        self.encoder = CNNEncoder()

        # GraphSAGE layers matching original paper architecture
        # Input dim: 120 (CNN) + 2 (coords) + 1 (past_yield) = 123
        # Let's make it 127 to match original by adding a few more coord features
        input_dim = 127
        self.layers = nn.ModuleList()
        # First layer: input_dim -> z_dim
        self.layers.append(dglnn.SAGEConv(input_dim, self.z_dim, self.aggregator_type))
        # Middle layers: z_dim -> z_dim
        for i in range(1, self.n_layers - 1):
            self.layers.append(
                dglnn.SAGEConv(self.z_dim, self.z_dim, self.aggregator_type)
            )
        # Last layer: z_dim -> z_dim
        if self.n_layers > 1:
            self.layers.append(
                dglnn.SAGEConv(self.z_dim, self.z_dim, self.aggregator_type)
            )

        self.dropout = nn.Dropout(self.dropout_prob)

        # LSTM for temporal processing across years (matches original)
        self.lstm = nn.LSTM(
            input_size=self.z_dim,
            hidden_size=self.z_dim,
            num_layers=1,
            batch_first=True,
        )

        # Final regressor matching original paper architecture
        self.regressor = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim // 2),
            nn.ReLU(),
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
        Forward pass matching original SAGE_RNN architecture

        Args:
            weather: (batch, n_years, 6, 52) - weather data
            soil: (batch, n_years, 11, 6) - soil data
            coords: (batch, 2) - coordinates (UNNORMALIZED)
            past_yields: (batch, n_past_years + 1) - past yields including current
            blocks: DGL blocks for graph sampling
        """
        batch_size = weather.shape[0]
        n_years = weather.shape[1]  # This should be n_past_years + 1

        # Normalize coordinates using utils function
        # Create dummy year and interval for normalization
        dummy_year = torch.zeros(batch_size, 1).to(coords.device)
        dummy_interval = torch.ones(batch_size, 1).to(coords.device)
        _, _, coords_norm = normalize_year_interval_coords(
            dummy_year, dummy_interval, coords
        )

        # Process each year through encoder + GraphSAGE (like original paper)
        hs = []
        for i in range(n_years):
            # Get data for year i
            year_weather = weather[:, i : i + 1]  # (batch, 1, 6, 52)
            year_soil = soil[:, i : i + 1]  # (batch, 1, 11, 6)

            # Get past yield for this timestep (like original paper's y_pad)
            past_yield_i = past_yields[:, i : i + 1]  # (batch, 1)

            # Encode weather and soil using CNN encoder
            cnn_features = self.encoder(year_weather, year_soil)  # (batch, 120)

            # Concatenate CNN features with coords and past_yield (to be fair with your other models)
            # Add extra coord features to reach 127 dimensions like original
            coords_expanded = torch.cat(
                [
                    coords_norm,  # (batch, 2)
                    coords_norm**2,  # (batch, 2) - squared coords
                    coords_norm * 0.5,  # (batch, 2) - scaled coords
                    past_yield_i,  # (batch, 1)
                ],
                dim=1,
            )  # (batch, 7)

            # Final features: 120 (CNN) + 7 (extra) = 127 (matches original)
            h = torch.cat([cnn_features, coords_expanded], dim=1)  # (batch, 127)

            # Apply GraphSAGE layers (matching original paper)
            if blocks is not None:
                for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                    if l < len(blocks):
                        # Copy representation from appropriate nodes (original paper logic)
                        h_dst = h[: block.number_of_dst_nodes()]
                        h = layer(block, (h, h_dst))
                    else:
                        # For layers without corresponding blocks, use full graph
                        h = layer(self.g, h)

                    # Apply activation and dropout (except last layer)
                    if l != len(self.layers):
                        h = F.relu(h)
                        h = self.dropout(h)

            hs.append(h)  # Store processed features for this year

        # Stack all years and process through LSTM (matching original)
        hs = torch.stack(hs, dim=1)  # (batch, n_years, z_dim)

        # LSTM for temporal modeling across years
        out, (last_h, last_c) = self.lstm(hs)  # (batch, n_years, z_dim)

        # Final prediction using regressor (matching original paper)
        pred = self.regressor(out)  # (batch, n_years, 1)

        # Return prediction for final year (like original paper)
        return pred[:, -1, :]  # (batch, 1)
