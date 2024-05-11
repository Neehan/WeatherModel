import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .constants import *
from . import utils


class FluDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(weather_path, flu_cases_path, n_past_weeks, n_predict_weeks):
    # Load the weather and the flu data
    flu_df = pd.read_json(flu_cases_path)
    flu_df = pd.DataFrame(flu_df.epidata.values.tolist())
    flu_df = flu_df[
        [
            "epiweek",
            "region",
        ]
        + FLU_DATASET_PARAMS
    ]
    flu_df["Year"] = flu_df["epiweek"] // 100
    flu_df["Week"] = flu_df["epiweek"] % 100

    flu_df = flu_df.drop(columns=["epiweek"])

    flu_df = flu_df.pivot(index=["Year", "region"], columns="Week")
    flu_df.columns = [f"{column}_{int(i)}" for column, i in flu_df.columns]
    flu_df.reset_index(drop=False, inplace=True)
    flu_df = flu_df.drop(
        columns=[column for column in flu_df.columns if column.endswith("_53")]
    )

    # keep only NYC cases
    flu_df = flu_df.loc[flu_df.region == "nyc"].sort_values(by=["Year"])

    weather_df = pd.read_csv(weather_path)
    weather_columns = [
        f"{varname}_{weekno}" for varname in WEATHER_PARAMS for weekno in range(1, 53)
    ]
    # Preprocess the weather data to get relevant columns
    weather_df = weather_df[
        ["Year", "lat", "lng", "region"] + weather_columns
    ].sort_values(by=["region", "Year"])

    weather_df["Year"] = weather_df.Year.astype(int)

    flu_cases_columns = [
        f"{column}_{i}" for column in FLU_DATASET_PARAMS for i in range(1, SEQ_LEN + 1)
    ]
    data_df = pd.merge(flu_df, weather_df, on=["region", "Year"])

    # standardize weather
    data_df = utils.standardize_data(
        data_df, WEATHER_PARAMS + FLU_DATASET_PARAMS, SEQ_LEN
    )

    num_regions = len(set(data_df["region"]))

    num_features = len(WEATHER_PARAMS) + len(FLU_DATASET_PARAMS)

    data_array = np.transpose(
        data_df[weather_columns + flu_cases_columns].values.reshape(
            (num_regions, -1, num_features, SEQ_LEN)
        ),
        (0, 1, 3, 2),  # num_regions x num_years x seq len x num input features
    ).reshape(
        num_regions, -1, num_features
    )  # num_regions x (num_years * seq len) x num input features

    # shift by 40 cause first 40 weeks of the first year are empty
    data_array = data_array[:, 39:, :]

    coords = data_df[["lat", "lng"]].values.reshape(num_regions, -1, 2)

    samples = []

    for region_id in range(data_array.shape[0]):
        for week_id in range(data_array.shape[1]):
            sample_values = np.zeros((n_past_weeks, num_features))
            mask = np.ones((n_past_weeks,)).astype(bool)
            sample_start = min(week_id + 1, n_past_weeks)

            sample_values[n_past_weeks - sample_start :, :] = data_array[
                region_id, week_id - sample_start + 1 : week_id + 1, :
            ]
            mask[n_past_weeks - sample_start :] = False
            ili_past = sample_values[:, -2]
            tot_cases_past = sample_values[:, -1]
            weather = sample_values[:, :-2]

            # target week is in future
            target_week_id = week_id + n_predict_weeks
            if target_week_id >= data_array.shape[1]:
                break

            ili_target = data_array[region_id, week_id + 1 : target_week_id + 1, -2]

            weather_index = (
                np.ones((n_past_weeks, 1)) * 7
            )  # used by weather transformer
            sample_coords = coords[region_id, 0, :]

            samples.append(
                (
                    weather.astype("float32"),
                    mask.astype("bool"),
                    weather_index.astype("int"),
                    sample_coords.astype("float32"),
                    ili_past.astype("float32"),
                    tot_cases_past.astype("float32"),
                    ili_target.astype("float32"),
                )
            )
    return samples


def train_test_split(
    weather_path,
    flu_cases_path,
    n_past_weeks=3,
    n_predict_weeks=1,
    test_year=2016,
    batch_size=64,
):
    dataset: list = load_data(
        weather_path, flu_cases_path, n_past_weeks, n_predict_weeks
    )
    dataset_size = len(dataset)  # Total number of items in the dataset
    test_start_idx = dataset_size - (2023 - test_year) * 52
    test_end_idx = test_start_idx + 52

    # Create Subset objects for train and test sets
    train_dataset = FluDataset(dataset[:test_start_idx])
    test_dataset = FluDataset(dataset[test_start_idx:test_end_idx])

    logging.debug(f"Training Dataset Size: {len(train_dataset)}")
    logging.debug(f"Test Dataset Size: {len(test_dataset)}")

    # Create the DataLoader for training and testing
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )  # Shuffling is False because data is time-dependent
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
