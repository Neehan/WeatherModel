from asyncio.log import logger
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.utils.constants import DRY_RUN, MAX_CONTEXT_LENGTH, TOTAL_WEATHER_VARS


class CropDataset(Dataset):
    def __init__(self, data, start_year, test_year, test_dataset=False, n_past_years=5):
        self.weather_cols = [f"W_{i}_{j}" for i in range(1, 7) for j in range(1, 53)]
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

        # Define weather indices used in preprocessing
        # 7: precipitation
        # 8: solar radiation
        # 11: snow depth
        # 1: max temp
        # 2: min temp
        # 29: vap pressure
        self.weather_indices = torch.tensor([7, 8, 11, 1, 2, 29])

        if test_dataset:  # test on specific year
            candidate_data = data[data["year"] == test_year]
        else:  # train on years from start_year to year before test_year
            candidate_data = data[
                (data["year"] >= start_year) & (data["year"] < test_year)
            ]

        # Filter to only include cases where we have complete historical data
        # Since there are no gaps in years, we just need to check if location
        # has been active for at least n_past_years + 1 years

        # For each location, find the first year it appears
        location_start_years = data.groupby("loc_ID")["year"].min()

        # For each candidate, check if (candidate_year - start_year + 1) >= n_past_years + 1
        # This is equivalent to: candidate_year >= start_year + n_past_years
        candidate_with_start = candidate_data.merge(
            location_start_years.rename("start_year"),
            left_on="loc_ID",
            right_index=True,
            how="left",
        )

        # Filter candidates that have enough historical data
        valid_candidate_data = candidate_with_start[
            candidate_with_start["year"]
            >= candidate_with_start["start_year"] + n_past_years
        ]

        self.index = valid_candidate_data[["year", "loc_ID"]].reset_index(drop=True)

        dataset_name = "train" if not test_dataset else "test"
        logger.info(
            f"Creating {dataset_name} dataloader with {len(self.index)} samples for {'test year ' + str(test_year) if test_dataset else 'training years ' + str(start_year) + '-' + str(test_year-1)}."
        )

        self.data = []
        total_samples = len(self.index)
        samples_to_process = total_samples // 20 if DRY_RUN else total_samples

        if total_samples == 0:
            logger.warning(f"No samples found for {dataset_name} dataset!")
            return

        for idx in range(min(samples_to_process, total_samples)):
            year, loc_ID = self.index.iloc[idx].values.astype("int")
            # Get exactly n_past_years + 1 years of data for this location
            query_data = data[(data["year"] <= year) & (data["loc_ID"] == loc_ID)].tail(
                n_past_years + 1
            )

            weather = (
                query_data[self.weather_cols]
                .values.astype("float32")
                .reshape((-1, 6, 52))
            )  # 6 measurements, 52 weeks
            practices = (
                query_data[self.practice_cols]
                .values.astype("float32")
                .reshape((-1, 14))
            )  # 14 practices
            soil = (
                query_data[self.soil_cols].values.astype("float32").reshape((-1, 11, 6))
            )  # 11 measurements, at 6 depths
            year_data = query_data["year"].values.astype("float32")
            coord = torch.FloatTensor(
                query_data[["lat", "lng"]].values.astype("float32")
            )

            # get the true yield
            y = query_data.iloc[-1:]["yield"].values.astype("float32").copy()
            y_past = query_data["yield"].values.astype("float32")
            if len(y_past) <= 1:
                raise ValueError(
                    f"Only 1 year of yield data for location {loc_ID} in year {year}. "
                    f"y_past value set to -5."
                )
            # the current year's yield is the target variable, so replace it with last year's yield
            y_past[-1] = y_past[-2]

            # Preprocess weather data for the model
            n_years, n_features, seq_len = weather.shape

            # Check context length constraint
            if n_years * seq_len > MAX_CONTEXT_LENGTH:
                raise ValueError(
                    f"n_years * seq_len = {n_years * seq_len} is greater than MAX_CONTEXT_LENGTH = {MAX_CONTEXT_LENGTH}"
                )

            # Transpose and reshape weather data: (n_years, n_features, seq_len) -> (n_years * seq_len, n_features)
            weather = weather.transpose(0, 2, 1)  # (n_years, seq_len, n_features)
            weather = weather.reshape(
                n_years * seq_len, n_features
            )  # (n_years * seq_len, n_features)

            # Process coordinates - use only the first coordinate (same for all years in this location)
            coord_processed = coord[0, :]  # (2,)

            # Expand year to match the sequence length
            # year_data is [n_years], need to add fraction for each week (1/52, 2/52, ..., 52/52)
            week_fractions = (
                torch.arange(1, seq_len + 1, dtype=torch.float32) / seq_len
            )  # [seq_len]
            year_expanded = torch.FloatTensor(year_data).unsqueeze(
                1
            ) + week_fractions.unsqueeze(  # [n_years, 1]
                0
            )  # [1, seq_len]  # [n_years, seq_len]
            year_expanded = year_expanded.contiguous().view(
                n_years * seq_len
            )  # [n_years * seq_len]

            # Create padded weather with specific weather indices
            padded_weather = torch.zeros(
                (seq_len * n_years, TOTAL_WEATHER_VARS),
            )
            padded_weather[:, self.weather_indices] = torch.FloatTensor(weather)

            # Create weather feature mask
            weather_feature_mask = torch.ones(
                TOTAL_WEATHER_VARS,
                dtype=torch.bool,
            )
            weather_feature_mask[self.weather_indices] = False
            weather_feature_mask = weather_feature_mask.unsqueeze(0).expand(
                n_years * seq_len, -1
            )

            # Create temporal interval (weekly data)
            interval = torch.full((1,), 7, dtype=torch.float32)

            self.data.append(
                (
                    padded_weather,  # (n_years * 52, TOTAL_WEATHER_VARS)
                    coord_processed,  # (2,)
                    year_expanded,  # (n_years * 52,)
                    interval,  # (1,)
                    weather_feature_mask,  # (n_years * 52, TOTAL_WEATHER_VARS)
                    practices,  # (n_years, 14)
                    soil,  # (n_years, 11, 6)
                    y_past,  # (n_years,)
                    y,  # (1,)
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def split_train_test_by_year(
    soybean_df: pd.DataFrame,
    n_train_years: int,
    test_year: int,
    standardize: bool = True,
    n_past_years: int = 5,
):
    # you need n_train_years + 1 years of data
    # n_train years to have at least one training datapoint
    # last 1 year is test year
    start_year = test_year - n_train_years

    data = soybean_df[
        soybean_df["year"] > 1981.0
    ]  # must be > 1981 otherwise all past data is just 0

    if standardize:
        cols_to_standardize = [
            col
            for col in data.columns
            if col not in ["loc_ID", "year", "State", "County", "lat", "lng", "yield"]
        ]

        # standardize the data
        data = pd.merge(
            data[["year", "State", "loc_ID", "lat", "lng", "yield"]],
            (data[cols_to_standardize] - data[cols_to_standardize].mean())
            / data[cols_to_standardize].std(),
            left_index=True,
            right_index=True,
        )
        # for yield always use same values so RMSEs are comparable across folds
        data["yield"] = (data["yield"] - 38.5) / 11.03

    data = data.fillna(0)

    train_dataset = CropDataset(
        data.copy(),
        start_year,
        test_year,
        test_dataset=False,
        n_past_years=n_past_years,
    )
    test_dataset = CropDataset(
        data.copy(), start_year, test_year, test_dataset=True, n_past_years=n_past_years
    )

    # Return the train and test datasets
    return train_dataset, test_dataset


def read_soybean_dataset(data_dir: str):
    full_filename = (
        "khaki_soybeans/soybean_data_soilgrid250_modified_states_9_processed.csv"
    )
    soybean_df = pd.read_csv(data_dir + full_filename)
    soybean_df = soybean_df.sort_values(["loc_ID", "year"])
    return soybean_df


def get_train_test_loaders(
    crop_df: pd.DataFrame,
    n_train_years: int,
    test_year: int,
    n_past_years: int,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:

    if n_train_years <= 1:
        raise ValueError(
            f"Not enough training data for current year + n_past_years. Required: {n_past_years + 1}. "
            f"Available training years: {n_train_years}."
        )

    if n_train_years < n_past_years + 1:
        logger.warning(
            f"Not enough training data for current year + n_past_years. Required: {n_past_years + 1}. "
            f"Available training years: {n_train_years}. "
            f"Setting n_past_years to {n_train_years - 1}."
        )
        n_past_years = n_train_years - 1

    train_dataset, test_dataset = split_train_test_by_year(
        crop_df,
        n_train_years,
        test_year,
        standardize=True,
        n_past_years=n_past_years,
    )

    if n_train_years < n_past_years + 1:
        raise ValueError(
            f"Not enough training data for current year + n_past_years. Required: {n_past_years + 1}. "
            f"Available training years: {n_train_years}."
        )

    train_loader = train_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = test_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader
