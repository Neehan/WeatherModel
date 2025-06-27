import logging
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.utils.constants import DRY_RUN, MAX_CONTEXT_LENGTH, TOTAL_WEATHER_VARS

logger = logging.getLogger(__name__)

# Global variables to store crop-specific scaling factors for RMSE conversion
CROP_SCALING_FACTORS = {}


class BaseDataProcessor(ABC):
    """Base class for data processing operations"""

    def __init__(self):
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
        self.weather_indices = torch.tensor([0, 1, 2, 4, 7, 8, 23, 30])

    def get_weather_columns(self) -> list:
        """Get all weather column names for 52 weeks"""
        weather_cols = []
        for var in self.weather_vars:
            for week in range(1, 53):
                weather_cols.append(f"{var}_{week}")
        return weather_cols

    def get_crop_yield_column(self, crop_type: str) -> str:
        """Get crop yield column name"""
        return f"{crop_type.lower().replace('winter', 'winter ')}_yield"


class WeatherDataProcessor(BaseDataProcessor):
    """Handles weather data aggregation and standardization"""

    def aggregate_weather_by_county(
        self, data: pd.DataFrame, crop_type: str
    ) -> pd.DataFrame:
        """Aggregate multiple weather records per county to single record"""
        crop_yield_col = self.get_crop_yield_column(crop_type)
        weather_cols = self.get_weather_columns()

        # Group by year and fips, then average coordinates and weather data
        agg_dict = {
            "state": "first",
            "county": "first",
            "lat": "mean",
            "lon": "mean",
            crop_yield_col: "first",  # Yield should be same for all records in county
        }

        # Add weather columns to aggregation
        for col in weather_cols:
            if col in data.columns:
                agg_dict[col] = "mean"

        aggregated = data.groupby(["year", "fips"]).agg(agg_dict).reset_index()

        logger.info(
            f"Aggregated {len(data)} records to {len(aggregated)} county-level records"
        )
        return aggregated

    def standardize_weather_data(
        self, data: pd.DataFrame, crop_type: str
    ) -> pd.DataFrame:
        """Standardize weather data and crop yields"""
        data = data.copy()
        weather_cols = self.get_weather_columns()

        # Weather parameter mappings for unit conversion
        weather_param_mapping = {
            0: {"unit_conversion": lambda x: x - 273.15},  # temp_avg: K to C
            1: {"unit_conversion": lambda x: x - 273.15},  # temp_max: K to C
            2: {"unit_conversion": lambda x: x - 273.15},  # temp_min: K to C
            3: {"unit_conversion": lambda x: x},  # precipitation: mm/day
            4: {"unit_conversion": lambda x: x},  # humidity: %
            5: {"unit_conversion": lambda x: x},  # wind_speed: m/s
            6: {
                "unit_conversion": lambda x: x * 24 * 3600 / 1e6
            },  # radiation: W/m² to MJ/m²/day
            7: {
                "unit_conversion": lambda x: x * 12.0
            },  # vpd: scale to match NASA range
        }

        # Process each weather variable
        for var_idx, var_name in enumerate(self.weather_vars):
            if var_idx not in weather_param_mapping:
                raise ValueError(
                    f"Weather variable index {var_idx} ({var_name}) not found in mapping"
                )

            unit_conversion = weather_param_mapping[var_idx]["unit_conversion"]

            for week in range(1, 53):
                col_name = f"{var_name}_{week}"
                if col_name not in data.columns:
                    continue

                # Apply unit conversion
                data[col_name] = unit_conversion(data[col_name])

                # Standardize
                mean_val = data[col_name].mean()
                std_val = data[col_name].std()

                if pd.isna(mean_val) or pd.isna(std_val):
                    raise ValueError(
                        f"Invalid statistics for column '{col_name}': mean={mean_val}, std={std_val}"
                    )

                if std_val > 0:
                    data[col_name] = (data[col_name] - mean_val) / std_val
                else:
                    logger.warning(f"Zero std for column '{col_name}', only centering")
                    data[col_name] = data[col_name] - mean_val

        # Standardize crop yield
        crop_yield_col = self.get_crop_yield_column(crop_type)
        if crop_yield_col in data.columns:
            crop_mean = data[crop_yield_col].mean()
            crop_std = data[crop_yield_col].std()

            CROP_SCALING_FACTORS[crop_type] = {"mean": crop_mean, "std": crop_std}
            print(
                f"CROP STATS - {crop_yield_col}: mean={crop_mean:.2f}, std={crop_std:.2f}"
            )

            data[crop_yield_col] = (data[crop_yield_col] - crop_mean) / crop_std
            logger.info(f"Standardized {crop_yield_col} using crop-specific scaling")

        # Fill NaN values
        data = data.fillna(0)
        return data


class CropNetDatasetBuilder(BaseDataProcessor):
    """Builds CropNet datasets with proper train/test handling"""

    def __init__(self, weather_processor: WeatherDataProcessor):
        super().__init__()
        self.weather_processor = weather_processor

    def filter_valid_locations(
        self, data: pd.DataFrame, crop_type: str, test_year: int
    ) -> pd.DataFrame:
        """Filter to only include counties with yield data in test year"""
        crop_yield_col = self.get_crop_yield_column(crop_type)

        test_year_data = data[data["year"] == test_year]
        valid_locations = test_year_data[test_year_data[crop_yield_col].notna()][
            "fips"
        ].unique()

        filtered_data = data[data["fips"].isin(valid_locations)]
        logger.info(
            f"Filtered to {len(valid_locations)} valid counties with {len(filtered_data)} records"
        )

        return filtered_data

    def prepare_data_for_dataset(
        self,
        data: pd.DataFrame,
        crop_type: str,
        start_year: int,
        test_year: int,
        is_test: bool,
        test_gap: int = 0,
    ) -> pd.DataFrame:
        """Prepare data for dataset creation"""
        # Filter by year range
        if is_test:
            candidate_data = data[data["year"] == test_year].copy()
        else:
            candidate_data = data[
                (data["year"] >= start_year) & (data["year"] < test_year - test_gap)
            ].copy()

        # For test dataset, aggregate multiple weather records per county
        if is_test:
            candidate_data = self.weather_processor.aggregate_weather_by_county(
                candidate_data, crop_type
            )

        return candidate_data

    def has_sufficient_history(
        self, data: pd.DataFrame, row: pd.Series, n_past_years: int
    ) -> bool:
        """Check if location has sufficient historical data"""
        year, fips = row["year"], row["fips"]
        loc_data = data[data["fips"] == fips]
        loc_data_up_to_year = loc_data[loc_data["year"] <= year]
        return len(loc_data_up_to_year.tail(n_past_years + 1)) == n_past_years + 1

    def create_sample_data(
        self, query_data: pd.DataFrame, crop_type: str, n_past_years: int
    ) -> tuple:
        """Create a single sample's data tensors"""
        crop_yield_col = self.get_crop_yield_column(crop_type)
        weather_cols = self.get_weather_columns()

        # Extract weather data (8 variables, 52 weeks)
        weather_data = query_data[weather_cols].values.astype("float32")
        weather = weather_data.reshape((-1, 8, 52))  # n_years x 8 variables x 52 weeks

        # Get coordinates and year data
        year_data = query_data["year"].values.astype("float32")
        coord = torch.FloatTensor(query_data[["lat", "lon"]].values.astype("float32"))

        # Get yields
        y = query_data.iloc[-1:][crop_yield_col].values.astype("float32").copy()
        y_past = query_data[crop_yield_col].values.astype("float32")

        if len(y_past) <= 1:
            raise ValueError(
                f"Only 1 year of yield data. y_past value set to previous year."
            )

        # Handle NaN values and replace current year yield with previous year
        y_past_series = pd.Series(y_past).ffill()
        y_past = y_past_series.values.astype("float32")
        y_past[-1] = y_past[-2]

        return self._process_weather_tensor(
            weather, year_data, coord, y_past, y, n_past_years
        )

    def _process_weather_tensor(
        self,
        weather: np.ndarray,
        year_data: np.ndarray,
        coord: torch.Tensor,
        y_past: np.ndarray,
        y: np.ndarray,
        n_past_years: int,
    ) -> tuple:
        """Process weather data into model format"""
        n_years, n_features, seq_len = weather.shape

        # Check context length constraint
        if n_years * seq_len > MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"n_years * seq_len = {n_years * seq_len} > MAX_CONTEXT_LENGTH = {MAX_CONTEXT_LENGTH}"
            )

        # Reshape weather data
        weather = weather.transpose(0, 2, 1)  # (n_years, seq_len, n_features)
        weather = weather.reshape(n_years * seq_len, n_features)

        # Process coordinates
        coord_processed = coord[0, :]  # Use first coordinate (same for all years)

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

        return (
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


class CropNetDataset(Dataset):
    """CropNet Dataset class"""

    def __init__(
        self,
        data: pd.DataFrame,
        crop_type: str,
        start_year: int,
        test_year: int,
        test_dataset: bool = False,
        n_past_years: int = 5,
        test_gap: int = 0,
    ):

        self.crop_type = crop_type
        self.weather_processor = WeatherDataProcessor()
        self.dataset_builder = CropNetDatasetBuilder(self.weather_processor)

        # Filter to valid locations and prepare data
        data = self.dataset_builder.filter_valid_locations(data, crop_type, test_year)
        candidate_data = self.dataset_builder.prepare_data_for_dataset(
            data, crop_type, start_year, test_year, test_dataset, test_gap
        )

        # Filter to locations with sufficient history
        data_sorted = data.sort_values(["fips", "year"])
        mask = candidate_data.apply(
            lambda row: self.dataset_builder.has_sufficient_history(
                data_sorted, row, n_past_years
            ),
            axis=1,
        )
        valid_candidates = candidate_data[mask]

        self.index = valid_candidates[["year", "fips"]].reset_index(drop=True)

        dataset_name = "test" if test_dataset else "train"
        year_info = (
            f"test year {test_year}"
            if test_dataset
            else f"training years {start_year}-{test_year-test_gap-1}"
        )
        logger.info(
            f"Creating {dataset_name} dataloader for {crop_type} with {len(self.index)} samples for {year_info}"
        )

        self._build_dataset(data_sorted, n_past_years)

    def _build_dataset(self, data: pd.DataFrame, n_past_years: int):
        """Build the dataset samples"""
        self.data = []
        total_samples = len(self.index)
        samples_to_process = total_samples // 20 if DRY_RUN else total_samples

        if total_samples == 0:
            logger.warning(f"No samples found for dataset!")
            return

        logger.info(f"Processing {samples_to_process} samples out of {total_samples}")

        for idx in range(min(samples_to_process, total_samples)):
            year, fips = self.index.iloc[idx].values.astype("int")

            # Get historical data for this location
            query_data = data[(data["year"] <= year) & (data["fips"] == fips)].tail(
                n_past_years + 1
            )

            try:
                sample_data = self.dataset_builder.create_sample_data(
                    query_data, self.crop_type, n_past_years
                )
                self.data.append(sample_data)
            except Exception as e:
                logger.warning(f"Failed to create sample for {fips} in {year}: {e}")
                continue

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
    standardize: bool = True,
    n_past_years: int = 5,
):
    """Split CropNet data into train/test by year for specific crop"""
    start_year = test_year - n_train_years
    data = cropnet_df[cropnet_df["year"] >= start_year].copy()

    # Forward fill missing yields within each location
    data = data.sort_values(["fips", "year"])
    crop_yield_col = f"{crop_type.lower().replace('winter', 'winter ')}_yield"
    data[crop_yield_col] = data.groupby("fips")[crop_yield_col].ffill()

    if standardize:
        weather_processor = WeatherDataProcessor()
        data = weather_processor.standardize_weather_data(data, crop_type)

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


def get_crop_rmse_conversion_factor(crop_type: str) -> float:
    """Get the RMSE conversion factor (std) for a specific crop type"""
    global CROP_SCALING_FACTORS

    if crop_type not in CROP_SCALING_FACTORS:
        raise ValueError(
            f"Crop scaling factors not found for {crop_type}. Available: {list(CROP_SCALING_FACTORS.keys())}"
        )

    return CROP_SCALING_FACTORS[crop_type]["std"]
