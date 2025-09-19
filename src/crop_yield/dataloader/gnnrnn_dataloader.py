import os
import pickle
import torch
import numpy as np
import pandas as pd
import dgl
import scipy.sparse as sp
import warnings
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Any, Optional
import math
import json
from asyncio.log import logger

from src.utils.constants import DATA_DIR, CROP_YIELD_STATS


def load_weather_scalers_from_json(json_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load weather parameter scalers from JSON file and convert to W_x_x format.

    Returns:
        Dictionary with keys like 'W_1_1', 'W_1_2', etc. and values containing 'mean' and 'std'
    """
    # Mapping from JSON parameter names to weather indices
    param_to_index = {
        "T2M_MAX": 1,  # max temp
        "T2M_MIN": 2,  # min temp
        "PRECTOTCORR": 7,  # precipitation
        "ALLSKY_SFC_SW_DWN": 8,  # solar radiation
        "SNODP": 11,  # snow depth
        "VAP": 29,  # vapor pressure
    }

    with open(json_path, "r") as f:
        scaler_data = json.load(f)

    weather_scalers = {}

    # Convert from JSON format to W_x_x format
    for week_key, week_data in scaler_data.items():
        if not week_key.startswith("week_"):
            continue

        week_num = int(week_key.split("_")[1])

        param_means = week_data["param_means"]
        param_stds = week_data["param_stds"]

        for param_name, weather_idx in param_to_index.items():
            if param_name in param_means and param_name in param_stds:
                col_name = f"W_{weather_idx}_{week_num}"
                weather_scalers[col_name] = {
                    "mean": param_means[param_name],
                    "std": param_stds[param_name],
                }

    return weather_scalers


def standardize_weather_cols_gnn(data: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Standardize only weather columns using either dataset-based or JSON-based scalers.

    Args:
        data: DataFrame containing weather data
        country: Non USA countries uses JSON scalers, others use dataset scalers)

    Returns:
        DataFrame with standardized weather columns only
    """
    data_copy = data.copy()

    # Get weather columns
    weather_cols = [f"W_{i}_{j}" for i in range(1, 7) for j in range(1, 53)]
    weather_cols_in_data = [col for col in weather_cols if col in data_copy.columns]

    if country.lower() != "usa":
        logger.warning("Using USA-based scalers for standardizing Non-US weather data")
        # Use JSON-based scalers for Argentina weather data only
        json_path = os.path.join(
            DATA_DIR, "khaki_soybeans", "weekly_weather_param_scalers.json"
        )
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON scalers file not found at {json_path}")

        weather_scalers = load_weather_scalers_from_json(json_path)

        # Standardize using JSON scalers
        for col in weather_cols_in_data:
            if col in weather_scalers:
                mean = weather_scalers[col]["mean"]
                std = weather_scalers[col]["std"]
                if std > 0:  # Avoid division by zero
                    data_copy[col] = (data_copy[col] - mean) / std
                else:
                    data_copy[col] = 0
    else:
        # Use dataset-based scalers for weather columns only
        if weather_cols_in_data:
            means = data_copy[weather_cols_in_data].mean()
            stds = data_copy[weather_cols_in_data].std()
            data_copy[weather_cols_in_data] = (
                data_copy[weather_cols_in_data] - means
            ) / stds
            # Fill any NaN values that result from division by zero with 0
            data_copy[weather_cols_in_data] = data_copy[weather_cols_in_data].fillna(0)

    return data_copy


def read_usa_dataset_gnn(data_dir: str):
    """Load USA dataset for GNN-RNN training"""
    full_filename = "khaki_soybeans/khaki_multi_crop_yield.csv"
    usa_df = pd.read_csv(data_dir + full_filename)
    usa_df = usa_df.sort_values(["loc_ID", "year"])
    return usa_df


def read_non_us_dataset_gnn(data_dir: str, country: str):
    """Load non-USA dataset for GNN-RNN training"""
    full_filename = f"khaki_soybeans/khaki_{country}_multi_crop.csv"
    df = pd.read_csv(data_dir + full_filename)
    df = df.sort_values(["loc_ID", "year"])
    return df


class GNNRNNDataset:
    def __init__(
        self,
        crop_df: pd.DataFrame,
        us_adj_file: str,
        crop_id_to_fid: str,
        test_year: int,
        n_past_years: int = 5,
        batch_size: int = 64,
        device: torch.device = torch.device("cpu"),
        crop_type: str = "soybean",
        test_dataset: bool = False,
        start_year: Optional[int] = None,
        standardization_stats: Optional[Dict] = None,
        test_gap: int = 0,
    ):
        self.crop_df = crop_df
        self.us_adj_file = us_adj_file
        self.crop_id_to_fid = crop_id_to_fid
        self.test_year = test_year
        self.n_past_years = n_past_years
        self.batch_size = batch_size
        self.device = device
        self.crop_type = crop_type
        self.test_dataset = test_dataset
        self.start_year = start_year
        self.standardization_stats = standardization_stats
        self.test_gap = test_gap

        self.yield_col = f"{crop_type}_yield"

        # Define column names like regular yield dataloader
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

        # Load and process data
        self._load_data()
        self._build_graph()
        self._create_sequences()

    def _load_data(self):
        """Load and filter data from CSV"""
        print("Loading CSV data...")

        # Filter to years > 1981 (same as main yield dataloader)
        self.filtered_df = self.crop_df[self.crop_df["year"] > 1981.0].copy()

        # Subtract test gap from start year (same as yield dataloader)
        start_year_adjusted = (
            self.start_year - self.test_gap if self.start_year is not None else None
        )

        # Filter by test/train years
        if self.test_dataset:
            data = self.filtered_df[self.filtered_df["year"] == self.test_year].copy()
        else:
            data = self.filtered_df[
                (self.filtered_df["year"] >= start_year_adjusted)
                & (self.filtered_df["year"] < self.test_year - self.test_gap)
            ].copy()

        # Drop rows with missing yield values
        data = data.dropna(subset=[self.yield_col])  # type: ignore
        print(f"After filtering: {len(data)} samples")

        # Filter to only include locations with sufficient history
        def has_sufficient_history(row):
            year, loc_ID = row["year"], row["loc_ID"]
            loc_data = self.filtered_df[self.filtered_df["loc_ID"] == loc_ID]
            loc_data_up_to_year = loc_data[loc_data["year"] <= year]
            return (
                len(loc_data_up_to_year.tail(self.n_past_years + 1))  # type: ignore
                == self.n_past_years + 1
            )

        mask = data.apply(has_sufficient_history, axis=1)
        self.valid_data = data[mask].copy()
        print(f"After history filtering: {len(self.valid_data)} samples")

        # Get unique locations and years
        self.counties = sorted(list(set(self.valid_data["loc_ID"])))
        self.years = sorted(list(set(self.valid_data["year"])))
        self.min_year = min(self.years)
        self.max_year = max(self.years)

        print(f"Counties: {len(self.counties)}, Years: {self.min_year}-{self.max_year}")

        # Data should already be standardized by get_gnnrnn_dataloaders()

    # Standardization removed - now done upfront in get_gnnrnn_dataloaders()

    def _build_graph(self):
        """Build adjacency matrix and graph structure - simplified version"""
        print("Building graph structure...")

        # Create simple adjacency matrix based on geographical proximity
        # This is a simplified version - in practice you'd use actual adjacency data
        n_counties = len(self.counties)

        # For now, create a simple graph where each node connects to a few neighbors
        adj_matrix = np.zeros((n_counties, n_counties))

        # Add self-connections
        np.fill_diagonal(adj_matrix, 1)

        # Add connections to nearby nodes (simplified - just connect to next few nodes)
        for i in range(n_counties):
            for j in range(max(0, i - 2), min(n_counties, i + 3)):
                if i != j:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        # Create DGL graph
        sp_adj = sp.coo_matrix(adj_matrix)
        self.g = dgl.from_scipy(sp_adj).to(self.device)
        self.adj_matrix = adj_matrix

        # Create mapping from county to index
        self.county_to_idx = {county: idx for idx, county in enumerate(self.counties)}

        print(f"Built graph with {n_counties} nodes and {sp_adj.nnz} edges")

    def _create_sequences(self):
        """Create sequences for GNN-RNN training"""
        print("Creating sequences...")

        self.sequences = []

        for _, row in self.valid_data.iterrows():
            year = int(row["year"])
            loc_ID = int(row["loc_ID"])

            # Get historical data for this location
            # Use filtered data (years > 1981) to match main dataloader
            loc_data = (
                self.filtered_df[
                    (self.filtered_df["loc_ID"] == loc_ID)
                    & (self.filtered_df["year"] <= year)
                ]
                .tail(self.n_past_years + 1)  # type: ignore
                .copy()
            )

            if len(loc_data) != self.n_past_years + 1:
                continue

            # Extract features (data already standardized)
            weather = (
                loc_data[self.weather_cols].values.astype(np.float32).reshape(-1, 6, 52)  # type: ignore
            )
            soil = loc_data[self.soil_cols].values.astype(np.float32).reshape(-1, 11, 6)  # type: ignore
            yields = loc_data[self.yield_col].values.astype(np.float32)  # type: ignore
            coords = loc_data[["lat", "lng"]].values.astype(np.float32)  # type: ignore

            # Current year yield is target, handle y_past like main dataloader
            target_yield = yields[-1]
            past_yields = yields.copy()  # Start with all yields including current
            if len(past_yields) <= 1:
                raise ValueError(
                    f"Only 1 year of yield data for location {loc_ID} in year {year}"
                )
            # Replace current year's yield with previous year's yield (same as main dataloader)
            past_yields[-1] = past_yields[-2]

            # Get county index for graph
            county_idx = self.county_to_idx.get(loc_ID, 0)

            self.sequences.append(
                {
                    "weather": weather,  # (n_years, 6, 52) = (n_past_years + 1, 6, 52)
                    "soil": soil,  # (n_years, 11, 6) = (n_past_years + 1, 11, 6)
                    "past_yields": past_yields,  # (n_past_years + 1,) - matches main dataloader
                    "target_yield": target_yield,  # scalar (standardized)
                    "coords": coords[0],  # (2,) - use first coord since same location
                    "county_idx": county_idx,
                    "year": year,
                }
            )

        print(f"Created {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def get_nodeloader(self):
        """Create DGL NodeDataLoader for graph sampling exactly like original paper"""
        N = len(self.counties)
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [10, 10]
        )  # Exactly like original
        nodeloader = dgl.dataloading.DataLoader(
            self.g,
            torch.arange(N).to(self.device),
            sampler,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            device=self.device,
        )
        return nodeloader

    def get_node_to_samples_mapping(self):
        """Get mapping from node_idx to sample indices"""
        node_to_samples = {}  # node_idx -> [sample_indices]

        for i, item in enumerate(self.sequences):
            node_idx = item["county_idx"]  # This is already set in _create_sequences
            if node_idx not in node_to_samples:
                node_to_samples[node_idx] = []
            node_to_samples[node_idx].append(i)

        return node_to_samples


def get_gnnrnn_dataloaders(
    crop_df: pd.DataFrame,
    test_year: int,
    n_train_years: int,
    n_past_years: int = 5,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    crop_type: str = "soybean",
    country: str = "usa",
    us_adj_file: str = "us_adj.pkl",
    crop_id_to_fid: str = "crop_id_to_fid.pkl",
    test_gap: int = 0,
) -> Tuple[Any, Any, Any, Dict, Dict]:
    """
    Get GNN-RNN data loaders for train and test
    Follows same standardization approach as main yield dataloader

    Args:
        crop_df: DataFrame containing crop data
        test_year: Year to use for testing
        n_train_years: Number of training years
        n_past_years: Number of past years for sequences
        batch_size: Batch size for training
        device: Device to use for training
        crop_type: Type of crop (soybean, corn, etc.)
        country: Country for data loading (affects weather standardization)
        us_adj_file: Adjacency file path
        crop_id_to_fid: Crop ID to field ID mapping file
        test_gap: Gap between training and test years

    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        nodeloader: DGL NodeDataLoader for graph sampling
        train_node_mapping: Mapping from node_idx to train sample indices
        test_node_mapping: Mapping from node_idx to test sample indices
    """
    start_year = test_year - n_train_years  # Use specified number of training years

    # Parameter validation (same as main dataloader)
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

    # Filter years like main dataloader (must be > 1981 otherwise all past data is just 0)
    data = crop_df[crop_df["year"] > 1981.0].copy()

    yield_col = f"{crop_type}_yield"

    # Drop rows with missing yield values for the given crop before standardization
    rows_before = len(data)
    data = data.dropna(subset=[yield_col])  # type: ignore
    rows_after = len(data)
    rows_dropped = rows_before - rows_after

    if rows_dropped > 0:
        logger.warning(
            f"Dropped {rows_dropped} rows with missing {crop_type} yield values ({rows_before} -> {rows_after} rows)"
        )

    data = data.fillna(0)

    # First standardize weather data using country-specific approach (same as main dataloader)
    data = standardize_weather_cols_gnn(data, country)

    # Then standardize non-weather data using original approach
    cols_to_exclude = [
        "loc_ID",
        "year",
        "State",
        "County",
        "lat",
        "lng",
        yield_col,
    ]
    # Also exclude weather columns since we already standardized them
    weather_cols = [f"W_{i}_{j}" for i in range(1, 7) for j in range(1, 53)]
    cols_to_exclude.extend(weather_cols)

    cols_to_standardize = [
        col for col in data.columns if col not in cols_to_exclude
    ]

    # Standardize non-weather data (soil, practices, etc.)
    if cols_to_standardize:
        data[cols_to_standardize] = (
            data[cols_to_standardize] - data[cols_to_standardize].mean()
        ) / data[cols_to_standardize].std()
        # Fill any NaN values that result from division by zero with 0
        data[cols_to_standardize] = data[cols_to_standardize].fillna(0)

    # Save crop-specific yield statistics from constants (same as main dataloader)
    train_data = data[(data["year"] >= start_year) & (data["year"] < test_year)]
    yield_mean, yield_std = (
        train_data[yield_col].mean(),
        train_data[yield_col].std(),
    )
    data[yield_col] = (data[yield_col] - yield_mean) / yield_std
    logger.info(
        f"Saving mean ({yield_mean:.3f}) and std ({yield_std:.3f}) from training data for {crop_type}"
    )
    CROP_YIELD_STATS[crop_type]["mean"].append(yield_mean)
    CROP_YIELD_STATS[crop_type]["std"].append(yield_std)

    # Fill remaining NaN values
    data = data.fillna(0)

    # Create training dataset (data already standardized)
    train_dataset = GNNRNNDataset(
        crop_df=data.copy(),
        us_adj_file=us_adj_file,
        crop_id_to_fid=crop_id_to_fid,
        test_year=test_year,
        n_past_years=n_past_years,
        batch_size=batch_size,
        device=device,
        crop_type=crop_type,
        test_dataset=False,
        start_year=start_year,
        standardization_stats=None,  # Data already standardized
        test_gap=test_gap,
    )

    # Create test dataset (data already standardized)
    test_dataset = GNNRNNDataset(
        crop_df=data.copy(),
        us_adj_file=us_adj_file,
        crop_id_to_fid=crop_id_to_fid,
        test_year=test_year,
        n_past_years=n_past_years,
        batch_size=batch_size,
        device=device,
        crop_type=crop_type,
        test_dataset=True,
        start_year=start_year,
        standardization_stats=None,  # Data already standardized
        test_gap=test_gap,
    )

    nodeloader = train_dataset.get_nodeloader()

    # Get proper node-to-samples mappings for both datasets
    train_node_mapping = train_dataset.get_node_to_samples_mapping()
    test_node_mapping = test_dataset.get_node_to_samples_mapping()

    return (
        train_dataset,
        test_dataset,
        nodeloader,
        train_node_mapping,
        test_node_mapping,
    )
