import logging
from typing import Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.utils.constants import DRY_RUN, MAX_CONTEXT_LENGTH, TOTAL_WEATHER_VARS

logger = logging.getLogger(__name__)

# Global variables to store crop-specific scaling factors for RMSE conversion
CROP_SCALING_FACTORS = {}


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
        self.crop_yield_col = f"{crop_type.lower().replace('winter', 'winter ')}_yield"

        # Weather variables we use (8 variables, 52 weeks each)
        self.weather_vars = [
            "temp_avg",
            "temp_max",
            "temp_min",
            "precipitation",
            "humidity",
            "wind_speed",
            "radiation",
            "vpd",
        ]
        self.weather_cols = []
        for var in self.weather_vars:
            for week in range(1, 53):
                self.weather_cols.append(f"{var}_{week}")

        # Mapping to pretraining dataset indices (31 weather vars total)
        self.weather_indices = torch.tensor([0, 1, 2, 4, 7, 8, 23, 30])

        # Determine target year for prediction
        if test_dataset:
            target_year = test_year  # predict test_year
        else:
            target_year = test_year - 1  # predict year before test_year

        # First filter: only keep counties that have yield data for target year
        target_year_data = data[data["year"] == target_year]
        valid_counties = target_year_data[
            target_year_data[self.crop_yield_col].notna()
        ]["fips"].unique()
        data = data[data["fips"].isin(valid_counties)].copy()

        logger.info(
            f"Filtered to {len(valid_counties)} counties with yield data for target year {target_year}"
        )

        # Forward fill yields within each county to handle missing historical data
        data = data.sort_values(["fips", "year"])
        data[self.crop_yield_col] = data.groupby("fips")[self.crop_yield_col].ffill()

        # Check for any remaining missing data and warn
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            logger.warning(f"Missing data found in columns: {missing_cols}")

        # Handle test vs training dataset differently
        if test_dataset:
            # For test dataset: aggregate by county and year to average multiple weather stations
            # First, filter to relevant years (we need historical data for lookups)
            # Aggregate by county (fips) and year - averaging weather data from multiple stations
            agg_dict = {
                "state": "first",
                "county": "first",
                "lat": "mean",  # Average coordinates across weather stations
                "lon": "mean",
                self.crop_yield_col: "first",  # Yield is the same for all stations in a county
            }
            for col in self.weather_cols:
                if col in data.columns:
                    agg_dict[col] = "mean"  # Average weather across stations

            # Aggregate by county and year
            data = data.groupby(["year", "fips"]).agg(agg_dict).reset_index()
            # Use fips as location identifier for aggregated data
            data["loc_id"] = data["fips"]

            # Now filter to test year for candidates
            candidate_data = data[data["year"] == test_year].copy()
        else:
            # For training: keep duplicates (multiple weather stations per county)
            # This provides more training data as requested
            data = data.copy()
            data["loc_id"] = data["lat"].astype(str) + "_" + data["lon"].astype(str)

            # Filter by year range
            candidate_data = data[
                (data["year"] >= start_year) & (data["year"] < test_year - test_gap)
            ].copy()

        # Sort data for historical lookup
        data_sorted = data.sort_values(["loc_id", "year"])

        # Filter candidates to only those with sufficient historical data
        valid_candidates = []
        for _, row in candidate_data.iterrows():
            year, loc_id = row["year"], row["loc_id"]
            loc_data = data_sorted[data_sorted["loc_id"] == loc_id]
            loc_data_up_to_year = loc_data[loc_data["year"] <= year]
            if len(loc_data_up_to_year.tail(n_past_years + 1)) == n_past_years + 1:
                valid_candidates.append(row)

        if valid_candidates:
            valid_candidates = pd.DataFrame(valid_candidates)
            self.index = valid_candidates[["year", "loc_id"]].reset_index(drop=True)
        else:
            self.index = pd.DataFrame(columns=["year", "loc_id"])

        dataset_name = "test" if test_dataset else "train"
        year_info = (
            f"test year {test_year}"
            if test_dataset
            else f"training years {start_year}-{test_year-test_gap-1}"
        )

        self.data = []
        total_samples = len(self.index)
        samples_to_process = total_samples // 20 if DRY_RUN else total_samples

        if total_samples == 0:
            logger.warning(f"No samples found for {dataset_name} dataset!")
            return

        # Track skipped samples for debugging
        skipped_insufficient_yield = 0
        skipped_context_length = 0

        for idx in range(min(samples_to_process, total_samples)):
            year, loc_id = self.index.iloc[idx].values

            # Get historical data for this location
            query_data = data_sorted[
                (data_sorted["year"] <= year) & (data_sorted["loc_id"] == loc_id)
            ].tail(n_past_years + 1)

            if len(query_data) < n_past_years + 1:
                continue

            # Extract weather data (8 variables, 52 weeks)
            weather_data = query_data[self.weather_cols].values.astype("float32")
            weather = weather_data.reshape(
                (-1, 8, 52)
            )  # n_years x 8 variables x 52 weeks

            # Get coordinates and year data
            year_data = query_data["year"].values.astype("float32")

            # Get yields
            y = (
                query_data.iloc[-1:][self.crop_yield_col]
                .values.astype("float32")
                .copy()
            )
            y_past = query_data[self.crop_yield_col].values.astype("float32")

            if len(y_past) <= 1:
                logger.warning(
                    f"Only 1 year of yield data for {loc_id} in {year}, skipping"
                )
                skipped_insufficient_yield += 1
                continue

            # Handle NaN values and replace current year yield with previous year
            # this does not affect target year since that has already been filtered
            y_past_series = pd.Series(y_past).ffill().bfill()
            y_past = y_past_series.values.astype("float32")
            y_past[-1] = y_past[-2]

            # Process weather tensor
            n_years, n_features, seq_len = weather.shape

            if n_years * seq_len > MAX_CONTEXT_LENGTH:
                logger.warning(
                    f"Skipping sample: n_years * seq_len = {n_years * seq_len} > MAX_CONTEXT_LENGTH = {MAX_CONTEXT_LENGTH}"
                )
                skipped_context_length += 1
                continue

            # Reshape weather data
            weather = weather.transpose(0, 2, 1)  # (n_years, seq_len, n_features)
            weather = weather.reshape(n_years * seq_len, n_features)

            # Expand year to match sequence length
            week_fractions = torch.arange(1, seq_len + 1, dtype=torch.float32) / seq_len
            year_expanded = torch.FloatTensor(year_data).unsqueeze(
                1
            ) + week_fractions.unsqueeze(0)
            year_expanded = year_expanded.contiguous().view(n_years * seq_len)

            # Create padded weather with specific indices
            padded_weather = torch.zeros((seq_len * n_years, TOTAL_WEATHER_VARS))
            padded_weather[:, self.weather_indices] = torch.FloatTensor(weather)

            # Create weather feature mask
            weather_feature_mask = torch.ones(TOTAL_WEATHER_VARS, dtype=torch.bool)
            weather_feature_mask[self.weather_indices] = False
            weather_feature_mask = weather_feature_mask.unsqueeze(0).expand(
                n_years * seq_len, -1
            )

            # Create temporal interval and dummy data
            interval = torch.full((1,), 7, dtype=torch.float32)
            practices = torch.zeros((n_years, 14), dtype=torch.float32)
            soil = torch.zeros((n_years, 11, 6), dtype=torch.float32)

            # all coords for the same location multiple year
            coord = torch.FloatTensor(
                query_data[["lat", "lon"]].values.astype("float32")
            )
            coord_processed = coord[0, :]  # Use first coordinate

            self.data.append(
                (
                    padded_weather,
                    coord_processed,
                    year_expanded,
                    interval,
                    weather_feature_mask,
                    practices,
                    soil,
                    y_past,
                    y,
                )
            )

        # Log accurate dataset size information after processing
        actual_samples = len(self.data)
        total_skipped = skipped_insufficient_yield + skipped_context_length

        logger.info(
            f"Creating {dataset_name} dataloader for {crop_type} with {actual_samples} samples for {year_info}"
        )

        if total_skipped > 0:
            logger.info(
                f"Skipped {total_skipped} samples during processing: "
                f"{skipped_insufficient_yield} insufficient yield data, "
                f"{skipped_context_length} context length exceeded"
            )

        if samples_to_process < total_samples:
            logger.info(
                f"Processed {samples_to_process}/{total_samples} candidate samples (DRY_RUN mode)"
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


def read_cropnet_dataset(data_dir: str) -> pd.DataFrame:
    """Read the combined CropNet dataset"""
    full_filename = "CropNet/combined_cropnet_data.csv"
    cropnet_df = pd.read_csv(data_dir + full_filename)
    cropnet_df = cropnet_df.sort_values(["fips", "year"])
    return cropnet_df


def split_train_test_by_year(
    cropnet_df: pd.DataFrame,
    crop_type: str,
    n_train_years: int,
    test_year: int,
    n_past_years: int = 5,
):
    """Split CropNet data into train/test by year for specific crop"""
    start_year = test_year - n_train_years
    data = cropnet_df[cropnet_df["year"] >= start_year].copy()

    # Forward fill missing yields within each location
    crop_yield_col = f"{crop_type.lower().replace('winter', 'winter ')}_yield"
    data = data.sort_values(["fips", "year"])
    data[crop_yield_col] = data.groupby("fips")[crop_yield_col].ffill()

    # Standardize everything
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
    weather_cols = []
    for var in weather_vars:
        for week in range(1, 53):
            weather_cols.append(f"{var}_{week}")

    # Standardize weather columns
    for col in weather_cols:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

    # Standardize crop yield and store scaling factors
    if crop_yield_col in data.columns:
        crop_mean = data[crop_yield_col].mean()
        crop_std = data[crop_yield_col].std()
        CROP_SCALING_FACTORS[crop_type] = {"mean": crop_mean, "std": crop_std}
        print(
            f"CROP STATS - {crop_yield_col}: mean={crop_mean:.2f}, std={crop_std:.2f}"
        )
        data[crop_yield_col] = (data[crop_yield_col] - crop_mean) / crop_std

    # Fill NaN values
    data = data.fillna(0)

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
            f"Not enough training data. Required: {n_past_years + 1}. Available: {n_train_years}"
        )

    if n_train_years < n_past_years + 1:
        logger.warning(
            f"Adjusting n_past_years from {n_past_years} to {n_train_years - 1}"
        )
        n_past_years = n_train_years - 1

    train_dataset, test_dataset = split_train_test_by_year(
        cropnet_df,
        crop_type,
        n_train_years,
        test_year,
        n_past_years=n_past_years,
    )

    train_loader = train_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = test_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader


def get_crop_rmse_conversion_factor(crop_type: str) -> float:
    """Get the RMSE conversion factor (std) for a specific crop type"""
    global CROP_SCALING_FACTORS

    if crop_type not in CROP_SCALING_FACTORS:
        raise ValueError(
            f"Crop scaling factors not found for {crop_type}. Available: {list(CROP_SCALING_FACTORS.keys())}"
        )

    return CROP_SCALING_FACTORS[crop_type]["std"]
