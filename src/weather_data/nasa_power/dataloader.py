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
    "usa_pt1",
    "usa_pt2",
    "southamerica_pt1",
    "southamerica_pt2",
    "southamerica_pt3",
]


def read_and_concatenate_csv(file_paths):
    """
    Read multiple CSV files and concatenate them into a single DataFrame.
    """
    weather_dfs = [pd.read_csv(file, index_col=False) for file in tqdm(file_paths)]
    return pd.concat(weather_dfs, ignore_index=True).drop_duplicates(
        subset=["Year", "lat", "lng"]
    )


def preprocess_data(weather_df, frequency, region, sequence_len):
    """
    Preprocess the DataFrame: drop duplicates and standardize columns.
    """
    print("Standardizing the columns.")
    if frequency == "weekly" and region == "USA":
        print("recreating means and stds for normalization.")
        param_means = dict()
        param_stds = dict()
    else:
        with open(DATA_DIR + f"/processed/weather_param_scalers.json", "r") as f:
            scalers = json.load(f)
            param_means, param_stds = scalers["param_means"], scalers["param_stds"]
            f.close()

    for param in tqdm(WEATHER_PARAMS):
        weather_cols = [f"{param}_{i}" for i in range(1, sequence_len + 1)]
        if frequency == "weekly" and region == "USA":
            param_mean = weather_df[weather_cols].values.mean()
            param_std = weather_df[weather_cols].values.std()
            param_means[param] = param_mean
            param_stds[param] = param_std
        else:
            param_mean = param_means[param]
            param_std = param_stds[param]

        weather_df[weather_cols] = (weather_df[weather_cols] - param_mean) / param_std

    with open(DATA_DIR + f"/processed/weather_param_scalers.json", "w") as f:
        json.dump({"param_means": param_means, "param_stds": param_stds}, f)
        f.close()

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

    # print("weather tensor shape: ", weather_tensor.shape)
    # print("coords tensor shape: ", coords_tensor.shape)
    # print("index tensor shape: ", index_tensor.shape)

    return TensorDataset(weather_tensor, coords_tensor, index_tensor)


def save_dataset(train_dataset, file_suffix, part_id, file_path):
    """
    Save the DataLoaders for future use.
    """
    torch.save(
        train_dataset,
        file_path + f"/processed/weather_dataset_{file_suffix}_{part_id}.pth",
    )


if __name__ == "__main__":

    for frequency in TIME_PARAMS.keys():
        frequency_days = TIME_PARAMS[frequency]["frequency"]
        sequence_len = TIME_PARAMS[frequency]["sequence_length"]
        file_suffix = frequency

        print(f"processing {frequency} weather data")

        offset = 0

        for region_id in REGIONS:
            region = region_id.split("_")[0].upper()
            region_coords = GRID[region]
            region_names = [f"{region.lower()}_{i}" for i in range(len(region_coords))]

            # num_regions = len(region_names)
            # input_paths = [
            #     f"{DATA_DIR}/{region}_regional_{FILE_SUFFIX}.csv"
            #     for region in region_names[2 * (num_regions // 3 + 1) :]
            # ]
            # weather_df = read_and_concatenate_csv(input_paths)
            # print("saving the merged dataset")
            # weather_df.to_csv(DATA_DIR + f"/{region.lower()}_weather_{FILE_SUFFIX}_pt3.csv")

            print(f"reading weather datasets for {region_id.upper()}")
            if (
                frequency == "daily"
                or frequency == "weekly"
                and region == "SOUTHAMERICA"
            ):
                filepart = "_" + region_id.split("_")[1]
            else:
                filepart = ""
                if int(region_id.split("_")[1][-1]) > 1:
                    continue
            weather_df = pd.read_csv(
                DATA_DIR
                + f"/csvs/{region.lower()}_weather_{file_suffix}{filepart}.csv",
                index_col=[0],
            )

            print("preprocessing.")
            weather_df = preprocess_data(weather_df, frequency, region, sequence_len)

            assert (
                len(weather_df) % NUM_YEARS == 0
            ), "dataset length is not divisible by number of years"
            dataset_length = len(weather_df)

            print("creating dataset.")
            chunk_length = 320 * NUM_YEARS
            for start_index in range(0, dataset_length, chunk_length):
                part_id = start_index // chunk_length + offset
                end_index = min(dataset_length, start_index + chunk_length)
                train_dataset = create_dataset(
                    weather_df.iloc[start_index:end_index], sequence_len, frequency_days
                )
                save_dataset(train_dataset, file_suffix, part_id, DATA_DIR)
                print("part_id: ", part_id)
            offset = part_id + 1
