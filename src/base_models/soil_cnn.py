import torch
import torch.nn as nn


class SoilCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.soil_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=8, out_channels=12, kernel_size=2, stride=1, padding=1
            ),
            # Flattening the output to fit Linear Layer
            nn.Flatten(),  # 24 x 1
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        self.soil_fc = nn.Sequential(
            nn.Linear(11 * 12, 40),
        )

    def forward(self, soil):
        batch_size, n_years = soil.shape[:2]

        # Reshape soil data for CNN processing
        soil = soil.reshape(batch_size * n_years * soil.shape[2], 1, -1)
        soil_out = self.soil_cnn(soil)
        soil_out = soil_out.view(batch_size * n_years, -1)
        soil_out = self.soil_fc(soil_out)
        soil_out = soil_out.view(batch_size, n_years, -1)

        return soil_out
