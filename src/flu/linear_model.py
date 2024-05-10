from .constants import *

import torch
import torch.nn as nn


class LinearFluPredictor(nn.Module):
    def __init__(self, input_length, output_length):
        super().__init__()
        # Fully connected layer to output the predicted flu cases
        self.fc = nn.Linear(input_length, output_length)

    def forward(
        self,
        weather,
        mask,
        weather_index,
        coords,
        ili_past,
        tot_cases_past,
    ):
        # Concatenate processed weather, last year's same week flu cases, and last week's flu cases
        combined_input = torch.cat(
            [
                weather,
                ili_past.unsqueeze(2),
                tot_cases_past.unsqueeze(2),
                # coords.unsqueeze(1).expand(-1, weather.shape[1], -1),
            ],
            dim=2,
        )
        output = self.fc(combined_input.view(combined_input.shape[0], -1))
        return output  # .squeeze()
