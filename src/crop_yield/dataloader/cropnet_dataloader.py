import json
import logging
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.utils.constants import DRY_RUN, MAX_CONTEXT_LENGTH, TOTAL_WEATHER_VARS

logger = logging.getLogger(__name__)


class CropNetDataset(Dataset):
    def __init__(
        self,
        data,
        crop_type,
        start_year,
        test_year,
        test_dataset=False,
        n_past_years=5,
        test_gap=0,
    ):
        self.crop_type = crop_type

        # Map CropNet weather columns to model indices
        # CropNet has 8 weather variables for 52 weeks each
        # temp_avg, temp_max, temp_min, precipitation, humidity, wind_speed, radiation, vpd
        self.cropnet_weather_cols = []
        weather_vars = [
            "temp_avg",
            "temp_max",
            "temp_min",
            "precipitation",
            "humidity",
            "wind_speed",
            "radiation",
            "vpd",
        ]

        for var in weather_vars:
            for week in range(1, 53):  # weeks 1-52
                self.cropnet_weather_cols.append(f"{var}_{week}")

        # Map CropNet weather variables to WeatherFormer indices
        # Based on preprocessing in yield_dataloader.py:
        # 7: precipitation, 8: solar radiation, 11: snow depth, 1: max temp, 2: min temp, 29: vap pressure
        self.weather_var_mapping = {
            "temp_max": 1,  # max temp
            "temp_min": 2,  # min temp
            "precipitation": 7,  # precipitation
            "radiation": 8,  # solar radiation (downward shortwave)
            "vpd": 29,  # vapor pressure deficit
            "temp_avg": 0,  # use index 0 for average temp (not in original mapping)
        }

        # Get crop yield column name
        crop_yield_col = f"{crop_type.lower().replace('winter', 'winter ')}_yield"

        if test_dataset:  # test on specific year
            candidate_data = data[data["year"] == test_year]
        else:  # train on years from start_year to year before test_year - test_gap
            candidate_data = data[
                (data["year"] >= start_year) & (data["year"] < test_year - test_gap)
            ]

        # Filter to only include counties that have yield data for this crop in test year (2021)
        test_year_data = data[data["year"] == test_year]
        valid_locations = test_year_data[test_year_data[crop_yield_col].notna()][
            "fips"
        ].unique()

        # Filter candidate data to only these valid locations
        candidate_data = candidate_data[candidate_data["fips"].isin(valid_locations)]

        # Filter to only include cases where we have complete historical data
        data_sorted = data.sort_values(["fips", "year"])

        def has_sufficient_history(row):
            year, fips = row["year"], row["fips"]
            loc_data = data_sorted[data_sorted["fips"] == fips]
            loc_data_up_to_year = loc_data[loc_data["year"] <= year]
            return len(loc_data_up_to_year.tail(n_past_years + 1)) == n_past_years + 1

        # Apply vectorized check
        mask = candidate_data.apply(has_sufficient_history, axis=1)
        valid_candidates = candidate_data[mask]

        self.index = valid_candidates[["year", "fips"]].reset_index(drop=True)

        dataset_name = "train" if not test_dataset else "test"
        logger.info(
            f"Creating {dataset_name} dataloader for {crop_type} with {len(self.index)} samples for {'test year ' + str(test_year) if test_dataset else 'training years ' + str(start_year) + '-' + str(test_year-test_gap-1)}."
        )

        self.data = []
        total_samples = len(self.index)
        samples_to_process = total_samples // 20 if DRY_RUN else total_samples

        if total_samples == 0:
            logger.warning(
                f"No samples found for {dataset_name} dataset for {crop_type}!"
            )
            return

        # Debug: Print some sample data statistics
        logger.info(f"Processing {samples_to_process} samples out of {total_samples}")

        for idx in range(min(samples_to_process, total_samples)):
            year, fips = self.index.iloc[idx].values.astype("int")
            # Get exactly n_past_years + 1 years of data for this location
            query_data = data[(data["year"] <= year) & (data["fips"] == fips)].tail(
                n_past_years + 1
            )

            # Extract weather data (8 variables, 52 weeks)
            weather_data = query_data[self.cropnet_weather_cols].values.astype(
                "float32"
            )
            weather = weather_data.reshape(
                (-1, 8, 52)
            )  # n_years x 8 variables x 52 weeks

            # Get coordinates and year data
            year_data = query_data["year"].values.astype("float32")
            coord = torch.FloatTensor(
                query_data[["lat", "lon"]].values.astype("float32")
            )

            # Get the true yield for this crop
            y = query_data.iloc[-1:][crop_yield_col].values.astype("float32").copy()
            y_past = query_data[crop_yield_col].values.astype("float32")

            if len(y_past) <= 1:
                raise ValueError(
                    f"Only 1 year of yield data for location {fips} in year {year}. "
                    f"y_past value set to previous year."
                )

            # Handle any remaining NaN values in y_past with forward fill
            y_past_series = pd.Series(y_past).ffill()
            y_past = y_past_series.values.astype("float32")

            # Replace current year yield with previous year yield for y_past
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

            # Map CropNet weather variables to model indices
            for cropnet_idx, var_name in enumerate(weather_vars):
                if var_name in self.weather_var_mapping:
                    model_idx = self.weather_var_mapping[var_name]
                    padded_weather[:, model_idx] = torch.FloatTensor(
                        weather[:, cropnet_idx]
                    )

            # Create weather feature mask - mask out unused features
            weather_feature_mask = torch.ones(
                TOTAL_WEATHER_VARS,
                dtype=torch.bool,
            )
            # Unmask only the features we're using
            for var_name in self.weather_var_mapping:
                model_idx = self.weather_var_mapping[var_name]
                weather_feature_mask[model_idx] = False

            weather_feature_mask = weather_feature_mask.unsqueeze(0).expand(
                n_years * seq_len, -1
            )

            # Create temporal interval (weekly data)
            interval = torch.full((1,), 7, dtype=torch.float32)

            # For CropNet, we don't have practice and soil data, so create dummy data
            practices = torch.zeros((n_years, 14), dtype=torch.float32)  # 14 practices
            soil = torch.zeros(
                (n_years, 11, 6), dtype=torch.float32
            )  # 11 measurements, 6 depths

            self.data.append(
                (
                    padded_weather,  # (n_years * 52, TOTAL_WEATHER_VARS)
                    coord_processed,  # (2,)
                    year_expanded,  # (n_years * 52,)
                    interval,  # (1,)
                    weather_feature_mask,  # (n_years * 52, TOTAL_WEATHER_VARS)
                    practices,  # (n_years, 14) - dummy for CropNet
                    soil,  # (n_years, 11, 6) - dummy for CropNet
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


def read_cropnet_dataset(data_dir: str):
    """Read the combined CropNet dataset"""
    full_filename = "CropNet/combined_cropnet_data.csv"
    cropnet_df = pd.read_csv(data_dir + full_filename)
    cropnet_df = cropnet_df.sort_values(["fips", "year"])
    return cropnet_df


def standardize_cropnet_data(data: pd.DataFrame, crop_type: str, weather_scalers: dict):
    """Standardize CropNet data using weather scalers and crop yield mean"""
    data = data.copy()

    # Get crop yield column
    crop_yield_col = f"{crop_type.lower().replace('winter', 'winter ')}_yield"

    # Standardize weather columns using provided scalers
    weather_vars = [
        "temp_avg",
        "temp_max",
        "temp_min",
        "precipitation",
        "humidity",
        "wind_speed",
        "radiation",
        "vpd",
    ]

    # Map CropNet weather variables to NASA POWER parameter names for scaling
    weather_param_mapping = {
        "temp_avg": "T2M",
        "temp_max": "T2M_MAX",
        "temp_min": "T2M_MIN",
        "precipitation": "PRECTOTCORR",
        "humidity": "RH2M",
        "wind_speed": "WS2M",
        "radiation": "ALLSKY_SFC_SW_DWN",
        "vpd": "VPD",
    }

    # First, convert CropNet units to NASA POWER units
    for var in weather_vars:
        for week in range(1, 53):
            col_name = f"{var}_{week}"
            if col_name in data.columns:
                # Apply unit conversions to match NASA POWER data
                if var in ["temp_avg", "temp_max", "temp_min"]:
                    # Convert from Kelvin to Celsius
                    data[col_name] = data[col_name] - 273.15
                elif var == "radiation":
                    # Convert from W/m² to MJ/m²/day
                    # NASA POWER ALLSKY_SFC_SW_DWN is in MJ/m²/day
                    data[col_name] = (
                        data[col_name] * 24 * 3600 / 1e6
                    )  # W/m² to MJ/m²/day
                elif var == "vpd":
                    # CropNet VPD appears to be in different units than NASA POWER
                    # Scale to match NASA POWER VPD range (mean=1.84, std=1.09)
                    data[col_name] = (
                        data[col_name] * 12.0
                    )  # Scale factor to match NASA range
                # precipitation, humidity, wind_speed should already be in correct units

    # Now apply NASA POWER standardization (CRITICAL: use NASA means/stds for pretrained model)
    for var in weather_vars:
        if var in weather_param_mapping:
            param_name = weather_param_mapping[var]
            if (
                param_name in weather_scalers["param_means"]
                and param_name in weather_scalers["param_stds"]
            ):
                mean_val = weather_scalers["param_means"][param_name]
                std_val = weather_scalers["param_stds"][param_name]

                # Standardize all weeks using NASA POWER scalers
                for week in range(1, 53):
                    col_name = f"{var}_{week}"
                    if col_name in data.columns:
                        data[col_name] = (data[col_name] - mean_val) / std_val

    # Standardize crop yield using dataset mean
    if crop_yield_col in data.columns:
        crop_mean = data[crop_yield_col].mean()
        crop_std = data[crop_yield_col].std()
        print(
            f"CROP STATS - {crop_yield_col}: mean={crop_mean:.2f}, std={crop_std:.2f}"
        )
        data[crop_yield_col] = (data[crop_yield_col] - crop_mean) / crop_std
        logger.info(
            f"Standardized {crop_yield_col} using fixed scaling (mean={crop_mean:.0f}, std={crop_std:.0f})"
        )

    return data


def split_train_test_by_year(
    cropnet_df: pd.DataFrame,
    crop_type: str,
    n_train_years: int,
    test_year: int,
    standardize: bool = True,
    n_past_years: int = 5,
):
    """Split CropNet data into train/test by year for specific crop"""
    # Calculate start year
    start_year = test_year - n_train_years

    # Filter data to relevant years
    data = cropnet_df[cropnet_df["year"] >= start_year].copy()

    # Get crop yield column name
    crop_yield_col = f"{crop_type.lower().replace('winter', 'winter ')}_yield"

    # Filter to only include counties that have yield data for this crop in test year (2021) FIRST
    test_year_data = data[data["year"] == test_year]
    valid_locations = test_year_data[test_year_data[crop_yield_col].notna()][
        "fips"
    ].unique()

    # Filter data to only these valid locations
    data = data[data["fips"].isin(valid_locations)]

    # THEN forward fill missing yields within each location to handle gaps (but not test year)
    # Sort by fips and year to ensure proper forward fill
    data = data.sort_values(["fips", "year"])
    data[crop_yield_col] = data.groupby("fips")[crop_yield_col].ffill()

    logger.info(
        f"After filtering for {crop_type} with test year {test_year}: {len(valid_locations)} valid counties, {len(data)} total records"
    )

    if standardize:
        # Load weather parameter scalers
        with open(
            "src/weather_preprocessing/nasa_power/weather_param_scalers.json", "r"
        ) as f:
            weather_scalers = json.load(f)

        data = standardize_cropnet_data(data, crop_type, weather_scalers)

    # Fill NaN values with 0
    # data = data.fillna(0)

    train_dataset = CropNetDataset(
        data.copy(),
        crop_type,
        start_year,
        test_year,
        test_dataset=False,
        n_past_years=n_past_years,
    )
    test_dataset = CropNetDataset(
        data.copy(),
        crop_type,
        start_year,
        test_year,
        test_dataset=True,
        n_past_years=n_past_years,
    )

    return train_dataset, test_dataset


def get_cropnet_train_test_loaders(
    cropnet_df: pd.DataFrame,
    crop_type: str,
    n_train_years: int,
    test_year: int,
    n_past_years: int,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Get train and test data loaders for CropNet data"""

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
        cropnet_df,
        crop_type,
        n_train_years,
        test_year,
        standardize=True,
        n_past_years=n_past_years,
    )

    train_loader = train_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = test_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader
