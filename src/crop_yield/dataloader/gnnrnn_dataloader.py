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

from src.utils.constants import DATA_DIR, CROP_YIELD_STATS


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
        self.start_year = start_year or (
            test_year - 10
        )  # Default to 10 years of training data
        self.standardization_stats = standardization_stats

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

        # Filter by test/train years
        if self.test_dataset:
            data = self.filtered_df[self.filtered_df["year"] == self.test_year].copy()
        else:
            data = self.filtered_df[
                (self.filtered_df["year"] >= self.start_year)
                & (self.filtered_df["year"] < self.test_year)
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
        """Create DGL NodeDataLoader for graph sampling"""
        N = len(self.counties)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
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


def get_gnnrnn_dataloaders(
    crop_df: pd.DataFrame,
    test_year: int,
    n_past_years: int = 5,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    crop_type: str = "soybean",
    us_adj_file: str = "us_adj.pkl",
    crop_id_to_fid: str = "crop_id_to_fid.pkl",
) -> Tuple[Any, Any, Any]:
    """
    Get GNN-RNN data loaders for train and test
    Follows same standardization approach as main yield dataloader

    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        nodeloader: DGL NodeDataLoader for graph sampling
    """
    start_year = test_year - 10  # Use 10 years of training data

    # Standardize data upfront (same as main yield dataloader)
    data = crop_df[crop_df["year"] > 1981.0].copy()  # Filter years like main dataloader

    yield_col = f"{crop_type}_yield"
    # Drop rows with missing yield values
    rows_before = len(data)
    data = data.dropna(subset=[yield_col])  # type: ignore
    rows_after = len(data)
    if rows_before != rows_after:
        print(
            f"Dropped {rows_before - rows_after} rows with missing {yield_col} values"
        )

    # Standardize everything (same logic as main dataloader)
    cols_to_standardize = [
        col
        for col in data.columns
        if col
        not in [
            "loc_ID",
            "year",
            "State",
            "County",
            "lat",
            "lng",
            yield_col,
        ]
    ]

    # Standardize weather and soil features
    data[cols_to_standardize] = (
        data[cols_to_standardize] - data[cols_to_standardize].mean()
    ) / data[cols_to_standardize].std()
    data[cols_to_standardize] = data[cols_to_standardize].fillna(0)

    # Standardize yield and save stats
    train_data = data[(data["year"] >= start_year) & (data["year"] < test_year)]
    yield_mean, yield_std = (
        train_data[yield_col].mean(),
        train_data[yield_col].std(),
    )
    data[yield_col] = (data[yield_col] - yield_mean) / yield_std
    print(
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
    )

    nodeloader = train_dataset.get_nodeloader()

    return train_dataset, test_dataset, nodeloader
