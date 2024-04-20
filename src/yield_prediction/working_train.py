import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

torch.use_deterministic_algorithms(True)


import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt

plt.style.use("ggplot")

import json
import torch.nn as nn
from tqdm import tqdm


def read_soybean_dataset():
    root = "/data/khaki_soybeans/"
    full_filename = "soybean_data_soilgrid250_modified_states_9_processed.csv"
    soybean_df = pd.read_csv(root + full_filename)
    soybean_df["year_std"] = soybean_df["year"]

    planting_cols = []  # [f"P_{i}" for i in range(1, 15)]
    discarded_cols = planting_cols  # + ["loc_ID"]

    # Group the data by year and calculate the mean yield for each year
    yearly_means = soybean_df.groupby("year")["yield"].mean()

    # Add a new column to soybean_df containing the mean yield for each year
    soybean_df["yearly_mean_yield"] = soybean_df["year"].map(yearly_means)

    return soybean_df[[col for col in soybean_df.columns if col not in discarded_cols]]


soybean_df = read_soybean_dataset()
soybean_df = soybean_df.sort_values(["loc_ID", "year"])

N_YEARS = 6
BATCH_SIZE = 64

import torch
from torch.utils.data import Dataset, DataLoader
import math


class SoybeanDataset(Dataset):
    def __init__(self, data, test_dataset=False, n_past_years=5):
        engineered_weather_cols = [
            f"{param}_{i}" for i in range(1, 53) for param in ["VAP", "VPD", "ET0"]
        ]
        self.weather_cols = [
            f"W_{i}_{j}" for i in range(1, 7) for j in range(1, 53)
        ]  # + engineered_weather_cols
        self.practice_cols = [f"P_{i}" for i in range(1, 15)]
        soil_measurements = [
            "bdod",
            "cec",
            "cfvo",
            "clay",
            "nitrogen",
            "ocd",
            "ocs",
            "phh2o",
            "sand",
            "silt",
            "soc",
        ]
        soil_depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
        self.soil_cols = [
            f"{measure}_mean_{depth}"
            for measure in soil_measurements
            for depth in soil_depths
        ]
        if test_dataset:  # test on missouri and kansas
            # test only final year
            self.index = data[
                (data["State"] == "missouri") | (data["State"] == "kansas")
            ][["year", "loc_ID"]].reset_index(drop=True)
        else:
            self.index = data[
                (data["State"] != "missouri") & (data["State"] != "kansas")
            ][["year", "loc_ID"]].reset_index(drop=True)

        self.data = []
        loc_ids = set(list(zip(*(self.index.values.astype("int").tolist())))[1])

        for idx in tqdm(range(len(self.index))):
            year, loc_ID = self.index.iloc[idx].values.astype("int")
            # look up last n years of data for a location
            query_data_true = data[(data["year"] <= year) & (data["loc_ID"] == loc_ID)]
            query_data = query_data_true
            # attention mask for the transformer model
            mask = np.zeros((n_past_years + 1,)).astype(bool)
            # pad shorter history with zeros
            if len(query_data) < n_past_years + 1:
                zero_df = pd.DataFrame(
                    0,
                    index=range(n_past_years + 1 - len(query_data)),
                    columns=query_data.columns,
                )
                query_data_true = pd.concat([zero_df, query_data_true]).reset_index(
                    drop=True
                )
                query_data = pd.concat([zero_df, query_data]).reset_index(drop=True)
                # make attention mask zero at the padded rows
                mask[: len(zero_df)] = True
            elif len(query_data) > n_past_years + 1:
                query_data_true = query_data_true.tail(n_past_years + 1)
                query_data = query_data.tail(n_past_years + 1)

            weather = (
                query_data[self.weather_cols]
                .values.astype("float32")
                .reshape((-1, 6, 52))
            )  # 9 measurements, 52 weeks
            practices = (
                query_data[self.practice_cols]
                .values.astype("float32")
                .reshape((-1, 14))
            )  # 14 practices
            soil = (
                query_data[self.soil_cols].values.astype("float32").reshape((-1, 11, 6))
            )  # 11 measurements, at 6 depths
            year = query_data["year"].values.astype("float32")
            year = (year - 1970.0) / 100.0
            coord = torch.FloatTensor(
                query_data[["lat", "lng"]].values.astype("float32")
            )

            # get the true yield
            y = query_data_true.iloc[-1:]["yield"].values.astype("float32").copy()
            y_past = query_data_true["yield"].values.astype("float32")
            # the current year's yield the target variable, so replace it with last year's yield
            y_past[-1] = y_past[-2]

            self.data.append((weather, practices, soil, year, coord, y, y_past, mask))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.data[idx]  # weather, practices, soil, year, y, y_mean

    def get_data_loader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size, shuffle=False)


def split_train_test_by_year(
    soybean_df,
    standardize=True,
    n_past_years=5,
    denoise=True,
):
    data = soybean_df[
        soybean_df["year"] > 1981.0
    ]  # must be > 1981 otherwise all past data is just 0

    if standardize:
        cols_to_standardize = [
            col
            for col in data.columns
            if col not in ["loc_ID", "year", "State", "County", "lat", "lng"]
        ]
        data = pd.merge(
            data[["year", "State", "loc_ID", "lat", "lng"]],
            (data[cols_to_standardize] - data[cols_to_standardize].mean())
            / data[cols_to_standardize].std(),
            left_index=True,
            right_index=True,
        )
    data = data.fillna(0)

    train_dataset = SoybeanDataset(
        data.copy(), test_dataset=False, n_past_years=n_past_years
    )
    test_dataset = SoybeanDataset(
        data.copy(), test_dataset=True, n_past_years=n_past_years
    )

    # Return the train and test datasets
    return train_dataset, test_dataset


train_dataset, test_dataset = split_train_test_by_year(
    soybean_df, True, n_past_years=N_YEARS
)

train_loader = train_dataset.get_data_loader(batch_size=BATCH_SIZE)
test_loader = test_dataset.get_data_loader(batch_size=BATCH_SIZE)

for weather, practices, soil, year, coord, y, y_past, mask in train_loader:
    print(
        "\n", weather.shape
    )  # (batch size) X (# years) X (# measurements) X (52 weeks)
    # print(year, mask)
    break

import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_rmse(model, data_loader):
    model.eval()
    # Compute the RMSE on the training dataset
    mse_total = 0.0
    for weather, practices, soil, year, coord, y, y_past, mask in data_loader:
        weather = weather.to(device)
        soil = soil.to(device)
        year = year.to(device)
        practices = practices.to(device)
        coord = coord.to(device)
        y = y.to(device)
        y_past = y_past.to(device)
        mask = mask.to(device)

        # Forward pass
        outputs = model(weather, soil, practices, year, coord, y_past, mask)
        # Compute the mean squared error
        mse = F.mse_loss(outputs, y.to(device))

        # Accumulate the MSE over all batches
        mse_total += mse.item()

    # Compute the RMSE
    rmse = np.sqrt(mse_total / len(data_loader))
    return rmse


def plot_losses(losses, taskname):
    with open(f"losses_{taskname}.json", "w") as f:
        json.dump(losses, f)
    f.close()

    plt.plot(range(1, len(losses["train"]) + 1), losses["train"], label="Train")
    plt.plot(range(1, len(losses["val"]) + 1), losses["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(taskname)
    plt.legend()
    plt.show()


import torch.optim as optim

# Set the random seed to a fixed value
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()

        assert (
            dim_model % 4 == 0
        ), "dim_model should be divisible by 4 for separate encoding"
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(p=0.1)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model).to(DEVICE)
        positions_list = (
            torch.arange(0, max_len, dtype=torch.float).view(-1, 1).to(DEVICE)
        )  # 0, 1, 2, 3, 4, 5
        self.div_term = torch.exp(
            torch.arange(0, dim_model, 4).float() * (-math.log(10000.0)) / dim_model
        ).to(
            DEVICE
        )  # 10000^(2i/dim_model)

        # PE(pos, 4i) = sin(pos/10000^(4i/dim_model))
        pos_encoding[:, 0::4] = torch.sin(positions_list * self.div_term)
        # PE(pos, 4i + 1) = cos(pos/1000^(4i/dim_model))
        pos_encoding[:, 1::4] = torch.cos(positions_list * self.div_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(
        self, token_embedding: torch.tensor, coords: torch.tensor
    ) -> torch.tensor:

        batch_size, seq_len, d_model = token_embedding.shape
        latitude, longitude = coords[:, :1], coords[:, 1:]
        # Normalize latitude and longitude
        lat_norm = (latitude / 180.0) * math.pi
        lon_norm = (longitude / 180.0) * math.pi

        # Create geo encoding
        geo_pe = torch.zeros(batch_size, seq_len, d_model, device=DEVICE)

        geo_pe[:, :, 2::4] = torch.sin(lat_norm * self.div_term).unsqueeze(1)
        geo_pe[:, :, 3::4] = torch.cos(lon_norm * self.div_term).unsqueeze(1)

        # Add positional encoding to input
        token_embedding = (
            token_embedding + self.pos_encoding[:seq_len, :].unsqueeze(0) + geo_pe
        )
        return token_embedding


class TransformerModel(nn.Module):
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
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
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


import torch.optim as optim

# Set the random seed to a fixed value

SEQ_LEN = 52
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        time_frequency = (self.position_list[:seq_len, :] * self.div_term).unsqueeze(0)
        # encode time in 4k and 4k + 1
        custom_pe[:, :, 0::4] = torch.sin(time_frequency)
        custom_pe[:, :, 1::4] = torch.cos(time_frequency)

        # Add positional encoding to input
        token_embedding += custom_pe
        return token_embedding


class Weatherformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=8,
        num_layers=3,
        hidden_dim_factor=8,
        max_len=180,
    ):
        super(Weatherformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.input_scaler = nn.Embedding(num_embeddings=31, embedding_dim=input_dim)
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


import copy

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

USE_EMBEDDING = True


class CombinedModel(nn.Module):
    def __init__(self, n_years):
        super().__init__()
        hidden_dim = 32
        self.soil_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),  #
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
            # nn.ReLu()
        )

        if USE_EMBEDDING:
            self.weather_transformer = Weatherformer(31, 48)

            self.weather_fc = nn.Sequential(
                nn.Linear(48 * 52, 120),
                # nn.ReLu()
            )
        else:
            self.weather_fc = nn.Sequential(
                nn.Linear(6 * 52, 120),
                # nn.ReLu()
            )

        fc_dims = 120 + 40 + 14 + 1 + 1
        self.trend_transformer = TransformerModel(
            input_dim=fc_dims,
            output_dim=32,
            num_layers=3,
        )
        self.fc1 = nn.Linear(in_features=32, out_features=1)

    def forward(self, weather, soil, practices, year, coord, y_past, mask):
        # summary_tokens = torch.full((weather.size(0), weather.size(1), weather.size(2), 1), 3).to(DEVICE)
        # # Add summary token at the start of each weather sequence
        # weather = torch.cat([summary_tokens, weather], dim=3)
        batch_size, n_years, n_features, n_weeks = weather.size()
        weather = weather.view(batch_size * n_years, -1, n_features)

        coord = coord.view(batch_size * n_years, 2)
        year = year.view(batch_size * n_years, 1)
        # [7, 8, 11, 1, 2, 29] are the closest weather feature ids according to pretraining
        weather_indices = torch.tensor([7, 8, 11, 1, 2, 29])
        padded_weather = torch.zeros(
            (
                batch_size * n_years,
                self.weather_transformer.max_len,
                self.weather_transformer.input_dim,
            ),
            device=DEVICE,
        )
        padded_weather[:, -52:, weather_indices] = weather
        # create feature mask
        weather_feature_mask = torch.ones(
            self.weather_transformer.input_dim,
            dtype=torch.bool,
            device=DEVICE,
        )
        weather_feature_mask[weather_indices] = False

        # create padding mask
        padding_mask = torch.ones(
            (batch_size * n_years, self.weather_transformer.max_len),
            dtype=torch.bool,
            device=DEVICE,
        )
        padding_mask[:, -n_weeks:] = False

        # create temporal index
        temporal_gran = torch.full((batch_size * n_years, 1), 7, device=DEVICE)
        temporal_index = torch.cat([year, temporal_gran], dim=1)

        weather = self.weather_transformer(
            padded_weather,
            coord,
            temporal_index,
            weather_feature_mask=weather_feature_mask,
            src_key_padding_mask=padding_mask,
        )[:, -52:, :]

        weather = weather.view(batch_size * n_years, -1)
        weather = self.weather_fc(weather)
        weather = weather.view(batch_size, n_years, -1)

        soil = soil.reshape(batch_size * n_years * soil.shape[2], 1, -1)
        soil_out = self.soil_cnn(soil)
        soil_out = soil_out.view(batch_size * n_years, -1)
        soil_out = self.soil_fc(soil_out)
        soil_out = soil_out.view(batch_size, n_years, -1)

        combined = torch.cat(
            (
                weather,
                soil_out,
                practices,
                year.reshape(batch_size, n_years, 1),
                y_past.unsqueeze(2),
            ),
            dim=2,
        )
        combined = self.trend_transformer(
            combined, coord.view(batch_size, -1, 2)[:, -1, :], mask
        )
        out = self.fc1(combined)
        return out


# Warm-up and decay function
def lr_lambda(current_epoch, num_warmup_epochs=10, decay_factor=0.95):
    if current_epoch < num_warmup_epochs:
        # Linear warm-up
        return float(current_epoch) / float(max(1, num_warmup_epochs))
    else:
        # Exponential decay
        return decay_factor ** (current_epoch - num_warmup_epochs)


model = CombinedModel(n_years=N_YEARS)
# move the model to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0009)
# Example of setting up the scheduler with the optimizer
num_epochs = 40  # Total training epochs
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

best_test_rmse = 999

losses = {
    "train": [],
    "val": [],
}

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, (
        weather,
        practices,
        soil,
        year,
        coord,
        y,
        y_past,
        mask,
    ) in enumerate(tqdm(train_loader)):
        # Zero the gradients
        optimizer.zero_grad()
        weather = weather.to(device)
        soil = soil.to(device)
        year = year.to(device)
        practices = practices.to(device)
        coord = coord.to(device)
        y = y.to(device)
        y_past = y_past.to(device)
        mask = mask.to(device)

        # Forward pass
        outputs = model(weather, soil, practices, year, coord, y_past, mask)
        loss = criterion(outputs, y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    scheduler.step()
    running_loss /= len(train_loader)
    running_loss = np.sqrt(running_loss)
    losses["train"].append(running_loss)
    test_rmse = compute_rmse(model, test_loader)
    losses["val"].append(test_rmse)
    best_test_rmse = min(test_rmse, best_test_rmse)
    print("best test rmse: ", best_test_rmse)
    print("current test rmse: ", test_rmse)
    print("[%d / %d] loss: %.3f" % (epoch + 1, num_epochs, running_loss))
