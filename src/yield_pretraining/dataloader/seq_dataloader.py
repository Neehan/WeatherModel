import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import json
from src.utils.constants import DRY_RUN


class SeqDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        n_past_years: int,
        yield_stats: Optional[Dict] = None,
    ):
        """
        Dataset for corn yield sequence prediction.

        Args:
            data: DataFrame with columns [Year, State, County, Value]
            n_past_years: Number of past years to use for prediction
            yield_stats: Pre-computed yield statistics (mean, std)
        """
        self.data = data.copy()
        self.n_past_years = n_past_years

        # Create sequences by county
        self.sequences = self._create_sequences()

        # Limit sequences for DRY_RUN
        if DRY_RUN:
            total_sequences = len(self.sequences)
            sequences_to_keep = max(1, total_sequences // 20)  # At least 1 sequence
            self.sequences = self.sequences[:sequences_to_keep]

        # Setup yield standardization
        if yield_stats is None:
            self.yield_stats = {}
            if yield_stats is None:
                self._compute_yield_stats()
        else:
            self.yield_stats = yield_stats

        # Standardize yield data only
        self._standardize_yields()

    def _create_sequences(self) -> List[Dict]:
        """Create sequences of yield data for each county."""
        sequences = []

        # Group by state and county
        grouped = self.data.groupby(["State", "County"])

        for (state, county), group in grouped:
            if (
                len(group) < self.n_past_years + 1
            ):  # Need at least n_past_years + 1 data points
                continue

            # Sort by year
            group = group.sort_values("Year")
            years = group["Year"].values
            yields = group["Value"].values
            # Get lat/lng for this county (should be consistent across years)
            lat = group["lat"].iloc[0]
            lng = group["lng"].iloc[0]

            # Create sequences - for each possible target year, use exactly n_past_years historical points
            for i in range(self.n_past_years, len(years)):
                # Use exactly n_past_years historical points
                past_years = years[i - self.n_past_years : i]
                past_yields = yields[i - self.n_past_years : i]
                target_year = years[i]
                target_yield = yields[i]

                # Calculate periods between points
                periods = np.diff(past_years)
                if len(periods) == 0:
                    periods = np.array([1])  # Default period for single point

                sequences.append(
                    {
                        "state": state,
                        "county": county,
                        "past_years": past_years,
                        "past_yields": past_yields,
                        "periods": periods,
                        "target_year": target_year,
                        "target_yield": target_yield,
                        "lat": lat,
                        "lng": lng,
                    }
                )

        return sequences

    def _compute_yield_stats(self):
        """Compute yield standardization statistics."""
        all_yields = []

        for seq in self.sequences:
            all_yields.extend(seq["past_yields"])
            all_yields.append(seq["target_yield"])

        all_yields = np.array(all_yields)
        self.yield_stats = {"mean": np.mean(all_yields), "std": np.std(all_yields)}

    def _standardize_yields(self):
        """Standardize yield values only."""
        mean = self.yield_stats["mean"]
        std = self.yield_stats["std"]

        for seq in self.sequences:
            seq["past_yields_std"] = (seq["past_yields"] - mean) / std
            seq["target_yield_std"] = (seq["target_yield"] - mean) / std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return a sequence sample.
        Model expects: year, coords, period, past_yield
        Each should be [batch_size, seq_len] tensors except coords which is [batch_size, seq_len, 2]
        """
        seq = self.sequences[idx]
        seq_len = len(seq["past_years"])

        # Prepare input tensors - keep raw values for year, coords, period
        years = torch.tensor(seq["past_years"], dtype=torch.float32)
        # Use actual lat/lng coordinates for this county
        coords = torch.full((seq_len, 2), fill_value=0.0, dtype=torch.float32)
        coords[:, 0] = seq["lat"]  # latitude
        coords[:, 1] = seq["lng"]  # longitude

        # For periods, pad with 0 for first element
        periods_padded = np.concatenate([[0], seq["periods"]])[:seq_len]
        periods = torch.tensor(periods_padded, dtype=torch.float32)

        # Only standardize yields
        past_yields = torch.tensor(seq["past_yields_std"], dtype=torch.float32)

        target_yield = torch.tensor(seq["target_yield_std"], dtype=torch.float32)

        return years, coords, periods, past_yields, target_yield


class SeqDataloader:
    def __init__(
        self,
        crop_type: str,
        batch_size: int,
        test_year_cutoff: int,
        n_past_years: int,
    ):
        """
        Dataloader for crop yield sequence data.

        Args:
            crop_type: Type of crop (e.g., 'corn', 'winter_wheat')
            batch_size: Batch size for dataloaders
            test_year_cutoff: Years > this value go to test set
            n_past_years: Number of past years to use for each prediction
        """
        self.crop_type = crop_type
        self.data_path = f"data/USDA/{crop_type}_yield_processed.csv"
        self.batch_size = batch_size
        self.test_year_cutoff = test_year_cutoff
        self.n_past_years = n_past_years

        # Load data
        self.data = pd.read_csv(self.data_path)

        # Create train/test split by year
        self.train_data, self.test_data = self._create_train_test_split()

        # Create datasets once during initialization
        self.train_dataset = SeqDataset(
            self.train_data, n_past_years=self.n_past_years, yield_stats=None
        )

        # Create test dataset using train yield statistics
        self.test_dataset = SeqDataset(
            self.test_data,
            n_past_years=self.n_past_years,
            yield_stats=self.train_dataset.yield_stats,
        )

    def _create_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split by year."""
        train_data = self.data[self.data["Year"] <= self.test_year_cutoff]
        test_data = self.data[self.data["Year"] > self.test_year_cutoff]

        return train_data, test_data

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders using pre-created datasets."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, test_loader

    def get_yield_stats(self) -> Dict:
        """Get the computed yield statistics for denormalization."""
        return self.train_dataset.yield_stats
