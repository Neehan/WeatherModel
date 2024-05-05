import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .constants import *


class FluDataset(Dataset):
    def __init__(self, weather_path, flu_cases_path, history_weeks=52):
        self.history_weeks = history_weeks
        self.weather_columns = [
            f"{varname}_{weekno}"
            for varname in WEATHER_PARAMS
            for weekno in range(1, 53)
        ]
        self.dataframe = self.read_data(weather_path, flu_cases_path)

    def read_data(self, weather_path, flu_cases_path):
        # Load the weather and the flu data
        flu_df = pd.read_json(flu_cases_path)
        print(flu_df.head())

    #     flu_df = flu_df[
    #         [
    #             "epiweek",
    #             "region",
    #             "num_ili",
    #             "num_patients",
    #         ]
    #     ]
    #     flu_df["Year"] = flu_df["epiweek"] // 100
    #     flu_df["Week"] = flu_df["epiweek"] % 100

    #     flu_df = flu_df.drop(columns=["epiweek"])

    #     flu_df = flu_df.pivot(index=["Year", "region"], columns="Week")
    #     flu_df.columns = [f"{column}_{int(i)}" for column, i in flu_df.columns]
    #     flu_df.reset_index(drop=False, inplace=True)

    #     weather_df = pd.read_csv(weather_path)
    #     # Preprocess the weather data to get relevant columns
    #     weather_df = weather_df[
    #         ["Year", "lat", "lng", "region"] + self.weather_columns
    #     ].sort_values(by=["region", "Year"])

    #     weather_df["Year"] = weather_df.Year.astype(int)

    #     data_df = pd.merge(flu_df, weather_df, on=["region", "Year"], how="right")
    #     data_df = data_df.bfill()
    #     print(data_df.head())
    #     return data_df

    # def __len__(self):
    #     return len(self.dataframe) - self.history_weeks

    # def __getitem__(self, idx):
    #     start_idx = idx
    #     end_idx = idx + self.history_weeks

    #     # Features from past year's weather
    #     weather_features = self.dataframe.iloc[start_idx:end_idx][
    #         weather_columns
    #     ].values.flatten()

    #     # Target variable from flu data
    #     target = self.dataframe.iloc[end_idx]["num_ili"]

    #     # Past num_ili and num_patients as features
    #     past_ili_patients = self.dataframe.iloc[start_idx:end_idx][
    #         ["num_ili", "num_patients"]
    #     ].values.flatten()

    #     features = np.concatenate([weather_features, past_ili_patients])

    #     return torch.tensor(features, dtype=torch.float), torch.tensor(
    #         target, dtype=torch.float
    #     )


# # Usage
# dataset = FluDataset(merged_df)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Assuming use of CUDA if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Example of getting a batch
# for features, targets in dataloader:
#     features, targets = features.to(device), targets.to(device)
#     # proceed with training...


def read_data(weather_columns, weather_path, flu_cases_path, n_past_weeks=3):
    # Load the weather and the flu data
    flu_df = pd.read_json(flu_cases_path)
    flu_df = pd.DataFrame(flu_df.epidata.values.tolist())
    flu_df = flu_df[
        [
            "epiweek",
            "region",
            "num_ili",
            "num_patients",
        ]
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

    weather_df = pd.read_csv(weather_path)
    # Preprocess the weather data to get relevant columns
    weather_df = weather_df[
        ["Year", "lat", "lng", "region"] + weather_columns
    ].sort_values(by=["region", "Year"])

    weather_df["Year"] = weather_df.Year.astype(int)

    data_df = pd.merge(flu_df, weather_df, on=["region", "Year"], how="right")
    num_regions = len(set(data_df["region"]))

    flu_cases_columns = [
        f"{column}_{i}" for column in ["num_ili", "num_patients"] for i in range(1, 53)
    ]

    num_features = len(WEATHER_PARAMS) + 2

    data_array = np.transpose(
        data_df[weather_columns + flu_cases_columns].values.reshape(
            (num_regions, -1, num_features, SEQ_LEN)
        ),
        (0, 1, 3, 2),  # num_regions x num_years x seq len x num input features
    ).reshape(
        num_regions, -1, num_features
    )  # num_regions x (num_years * seq len) x num input features

    # shift by 40 cause first 40 weeks of the first year are empty
    data_array = data_array[:, 40:, :]

    coords = data_df[["lat", "lng"]].values.reshape(num_regions, -1, 2)

    samples = []

    for region_id in range(data_array.shape[0]):
        for week_id in range(1, data_array.shape[1]):
            sample_values = np.zeros((n_past_weeks, num_features))
            mask = np.ones((n_past_weeks,)).astype(bool)
            sample_start = min(week_id + 1, n_past_weeks)

            print(sample_start)
            print(sample_values[-sample_start:, :].shape)
            print("data entry", data_array.shape)

            sample_values[-sample_start:, :] = data_array[
                region_id, week_id - sample_start + 1 : week_id + 1, :
            ]
            mask[-sample_start:] = False
            ili_past = sample_values[:-1, -2]
            ili_target = sample_values[-1, -2]
            tot_cases_past = sample_values[:-1, -1]
            weather = sample_values[:, :-2]

            weather_index = (
                np.ones((n_past_weeks, 2)) * 7
            )  # used by weather transformer
            sample_coords = np.repeat(coords[region_id, :1, :], n_past_weeks, axis=0)

            samples.append(
                (
                    weather,
                    mask,
                    weather_index,
                    sample_coords,
                    ili_past,
                    tot_cases_past,
                    ili_target,
                )
            )

    return samples


weather_columns = [
    f"{varname}_{weekno}" for varname in WEATHER_PARAMS for weekno in range(1, 53)
]

weather_path = DATA_DIR + "weather_weekly.csv"
flu_cases_path = DATA_DIR + "flu_cases.json"

_ = read_data(weather_columns, weather_path, flu_cases_path)
