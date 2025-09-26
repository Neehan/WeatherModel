import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import json
from grids import GRID
from constants import *

MAX_LENGTH = 365

TIME_PARAMS = {
    # weekly must be first cause we save their stds
    "weekly": {
        "frequency": 7,
        "sequence_length": 52,
    },
    "daily": {
        "frequency": 1,
        "sequence_length": 365,
    },
    "monthly": {
        "frequency": 30,
        "sequence_length": 12,
    },
}

np.random.seed(1234)
torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

DATA_DIR = "data/nasa_power"
REGIONS = [
    f"{region.lower()}_{i}"
    for region, coords in GRID.items()
    for i, _ in enumerate(coords)
    if region == "CENTRALAMERICA"
]


def preprocess_data(weather_df, sequence_len):
    """
    Preprocess the DataFrame: drop duplicates and standardize columns.
    """
    print("Standardizing the columns.")

    with open(DATA_DIR + f"/processed/weather_param_scalers.json", "r") as f:
        scalers = json.load(f)
        param_means, param_stds = scalers["param_means"], scalers["param_stds"]
        f.close()

    for param in tqdm(WEATHER_PARAMS):
        weather_cols = [f"{param}_{i}" for i in range(1, sequence_len + 1)]
        param_mean = param_means[param]
        param_std = param_stds[param]

        weather_df[weather_cols] = (weather_df[weather_cols] - param_mean) / param_std

    print("Sorting by location and year")
    weather_df = weather_df.sort_values(by=["lat", "lng", "Year"])
    return weather_df


def create_dataset(weather_df, seq_len, frequency_days):
    weather_params = [
        f"{param}_{i}" for i in range(1, seq_len + 1) for param in WEATHER_PARAMS
    ]
    weather_measurements = np.transpose(
        weather_df[weather_params].values.reshape(
            (-1, NUM_YEARS, len(WEATHER_PARAMS), seq_len)
        ),
        # locs x years x seq len x num weather vars
        (0, 1, 3, 2),
    ).reshape(-1, NUM_YEARS * seq_len, len(WEATHER_PARAMS))

    # now reshape to have MAX_LENGTH
    num_segments = NUM_YEARS * seq_len // MAX_LENGTH
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
    # daily, weekly, monthly etc
    data_frequency = frequency_days * np.ones((num_regions, num_segments, 1))
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

    for frequency in TIME_PARAMS.keys():
        frequency_days = TIME_PARAMS[frequency]["frequency"]
        sequence_len = TIME_PARAMS[frequency]["sequence_length"]
        file_suffix = frequency

        print(f"processing {frequency} weather data")

        file_id = 109  # up to 33 is us and south am data

        for region_id in REGIONS:
            region_name = region_id.split("_")[0].upper()

            print(f"reading weather dataset for {region_id.upper()}")

            weather_df = pd.read_csv(
                f"{DATA_DIR}/csvs/{region_id}_regional_{file_suffix}.csv",
                index_col=False,
            )

            print("preprocessing.")
            weather_df = preprocess_data(weather_df, sequence_len)

            assert (
                len(weather_df) % NUM_YEARS == 0
            ), "dataset length is not divisible by number of years"
            dataset_length = len(weather_df)

            print("creating dataset.")
            train_dataset = create_dataset(weather_df, sequence_len, frequency_days)
            filename = f"/processed/weather_dataset_{file_suffix}_{file_id}.pt"
            save_dataset(train_dataset, filename, DATA_DIR)
            file_id = file_id + 1
