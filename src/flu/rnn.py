import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .constants import *


def move_padding_to_end(tensor, mask):
    """Move padding from the beginning to the end of the sequence."""
    max_len = tensor.size(1)
    new_tensor = torch.zeros_like(tensor)
    new_mask = torch.zeros_like(mask)
    for i in range(tensor.size(0)):
        seq_len = mask[i].sum().item()
        new_tensor[i, :seq_len] = tensor[i, mask[i] == 1]
        new_mask[i, :seq_len] = 1
    return new_tensor, new_mask


class LSTMFluPredictor(nn.Module):
    def __init__(self, n_predict_weeks=5, input_dim=1 + 2, num_layers=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(
            hidden_dim, n_predict_weeks
        )  # Match the output dimensions to n_predict_weeks

    def forward(self, weather, mask, weather_index, coords, ili_past, tot_cases_past):
        weather_indices = torch.tensor(
            [
                0,
                # 4,
                # 6,
                # 7,
                # 8,
                # 24, 25
            ],
            device=DEVICE,
            dtype=torch.int,
        )
        weather = weather[:, :, weather_indices]

        # Concatenate weather data with ili_past and tot_cases_past
        combined_input = torch.cat(
            [weather, ili_past.unsqueeze(-1), tot_cases_past.unsqueeze(-1)], dim=-1
        )

        # Move padding to the end
        combined_input, mask = move_padding_to_end(combined_input, mask)

        # Convert mask to lengths for packing
        lengths = (~mask).sum(
            dim=1
        )  # Assuming mask is binary (1 for valid, 0 for padded)

        # Packing the sequence
        packed_input = pack_padded_sequence(
            combined_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpacking the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Assuming we need the last valid output for each sequence
        idx = (lengths - 1).view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
        last_output = output.gather(1, idx).squeeze(1)

        # Passing through the fully connected layer
        predictions = self.fc(last_output)

        # Updating predictions for the first output week with the last available ili_past
        predictions[:, 0] += ili_past[:, -1]

        return predictions
