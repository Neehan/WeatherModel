import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import json
from grids import GRID
from constants import *

MAX_LENGTH = 365

# Only weekly processing
WEEKLY_SEQUENCE_LENGTH = 52
WEEKLY_FREQUENCY_DAYS = 7

np.random.seed(1234)
torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

DATA_DIR = "data/nasa_power"

REGION_MAP = {
    region.lower(): [f"{region.lower()}_{i}" for i, _ in enumerate(coords)]
    for region, coords in GRID.items()
}

print(REGION_MAP)


def preprocess_data_weekly_scalers(weather_df):
    """
    Preprocess the DataFrame using weekly-specific scalers for weekly data only.
    """

    # Load weekly scalers
    with open(DATA_DIR + f"/processed/weekly_weather_param_scalers.json", "r") as f:
        weekly_scalers = json.load(f)

    for param in WEATHER_PARAMS:
        for week in range(1, WEEKLY_SEQUENCE_LENGTH + 1):  # weeks 1-52
            weather_col = f"{param}_{week}"

            if weather_col in weather_df.columns:
                # Get weekly-specific mean and std
                week_key = f"week_{week}"
                if week_key in weekly_scalers:
                    param_mean = weekly_scalers[week_key]["param_means"][param]
                    param_std = weekly_scalers[week_key]["param_stds"][param]

                    # Apply weekly-specific standardization
                    if param_std > 0:  # Avoid division by zero
                        weather_df[weather_col] = (
                            weather_df[weather_col] - param_mean
                        ) / param_std
                    else:
                        print(
                            f"Warning: std=0 for {param} week {week}, skipping standardization"
                        )
                else:
                    print(f"Warning: No weekly scaler found for week {week}")

    weather_df = weather_df.sort_values(by=["lat", "lng", "Year"])
    return weather_df


def create_dataset(weather_df):
    """
    Create dataset for weekly data only.
    """
    weather_params = [
        f"{param}_{i}"
        for i in range(1, WEEKLY_SEQUENCE_LENGTH + 1)
        for param in WEATHER_PARAMS
    ]
    weather_measurements = np.transpose(
        weather_df[weather_params].values.reshape(
            (-1, NUM_YEARS, len(WEATHER_PARAMS), WEEKLY_SEQUENCE_LENGTH)
        ),
        # locs x years x seq len x num weather vars
        (0, 1, 3, 2),
    ).reshape(-1, NUM_YEARS * WEEKLY_SEQUENCE_LENGTH, len(WEATHER_PARAMS))

    # now reshape to have MAX_LENGTH
    num_segments = NUM_YEARS * WEEKLY_SEQUENCE_LENGTH // MAX_LENGTH
    weather_measurements = np.concatenate(
        (
            weather_measurements[:, : num_segments * MAX_LENGTH, :],
            weather_measurements[:, -MAX_LENGTH:, :],  # add the remaining data
        ),
        axis=1,
    )
    num_segments += 1  # we added one additional data point above
    weather_measurements = weather_measurements.reshape(
        -1, MAX_LENGTH, len(WEATHER_PARAMS)
    )

    coords = weather_df[["lat", "lng"]].values.reshape(-1, NUM_YEARS, 2)

    num_regions = coords.shape[0]

    coords = np.repeat(coords[:, :1, :], num_segments, axis=1).reshape(-1, 2)

    # we need an index
    index = np.repeat(np.arange(num_segments)[None, :, None], num_regions, axis=0)
    # weekly frequency
    data_frequency = WEEKLY_FREQUENCY_DAYS * np.ones((num_regions, num_segments, 1))
    index = np.stack([index, data_frequency], axis=2).reshape(-1, 2)

    # Converting to tensors
    weather_tensor = torch.tensor(weather_measurements, dtype=torch.float32)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    index_tensor = torch.tensor(index, dtype=torch.float32)

    return TensorDataset(weather_tensor, coords_tensor, index_tensor)


def save_dataset(train_dataset, filename, file_path):
    """
    Save the DataLoaders for future use.
    """
    torch.save(
        train_dataset,
        file_path + filename,
    )


if __name__ == "__main__":
    print("Processing weekly weather data with weekly scalers")

    file_id = 0  # up to 33 is us and south am data

    for region, region_ids in REGION_MAP.items():
        for region_id in tqdm(region_ids, desc=f"Processing {region.upper()}"):
            filepath = f"{DATA_DIR}/csvs/{region}/{region_id}_regional_weekly.csv"
            weather_df = pd.read_csv(filepath, index_col=False)
            weather_df = preprocess_data_weekly_scalers(weather_df)

            assert (
                len(weather_df) % NUM_YEARS == 0
            ), "dataset length is not divisible by number of years"
            train_dataset = create_dataset(weather_df)
            filename = f"/processed/weather_dataset_weekly_{file_id}.pt"
            save_dataset(train_dataset, filename, DATA_DIR)
            file_id = file_id + 1

    print("Weekly scalers processing completed!")
