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
        known_years: int,
        total_years: int,
        yield_stats: Optional[Dict] = None,
    ):
        """
        Dataset for corn yield sequence prediction with masked yields.

        Args:
            data: DataFrame with columns [Year, State, County, Value]
            known_years: Number of known datapoints in the small history
            total_years: Total sequence length after backward expansion
            yield_stats: Pre-computed yield statistics (mean, std)
        """
        self.data = data.copy()
        self.known_years = known_years
        self.total_years = total_years

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
            self._compute_yield_stats()
        else:
            self.yield_stats = yield_stats

        # Standardize yield data
        self._standardize_yields()

    def _create_sequences(self) -> List[Dict]:
        """Create sequences of yield data for each county with masking."""
        sequences = []

        # Group by state and county
        grouped = self.data.groupby(["State", "County"])

        for (state, county), group in grouped:
            if len(group) < self.total_years:  # Need at least total_years data points
                continue

            # Sort by year
            group = group.sort_values("Year")
            years = group["Year"].values
            yields = group["Value"].values
            # Get lat/lng for this county (should be consistent across years)
            lat = group["lat"].iloc[0]
            lng = group["lng"].iloc[0]

            # Create sequences - for each possible set of total_years consecutive points
            for i in range(len(years) - self.total_years + 1):
                # Take total_years consecutive points
                all_years = years[i : i + self.total_years]
                all_yields = yields[i : i + self.total_years]

                # Create full sequence with known and unknown yields
                full_yields = np.zeros(self.total_years)
                mask = np.zeros(
                    self.total_years, dtype=bool
                )  # True for known positions

                # Place known yields at the END of the sequence (recent years)
                known_positions = np.arange(
                    self.total_years - self.known_years, self.total_years
                )
                known_yields = all_yields[
                    -self.known_years :
                ]  # Last known_years yields
                full_yields[known_positions] = known_yields
                mask[known_positions] = True

                # Fill in the actual yields for the unknown positions (historical years for loss computation)
                unknown_positions = np.arange(0, self.total_years - self.known_years)
                unknown_yields = all_yields[
                    : self.total_years - self.known_years
                ]  # First (total_years - known_years) yields
                full_yields[unknown_positions] = unknown_yields

                # Calculate periods between consecutive years (all 1 for yearly data)
                periods = np.ones(self.total_years - 1)

                sequences.append(
                    {
                        "state": state,
                        "county": county,
                        "years": all_years,
                        "yields": full_yields,
                        "mask": mask,  # True for known yields, False for masked yields
                        "periods": periods,
                        "lat": lat,
                        "lng": lng,
                    }
                )

        return sequences

    def _compute_yield_stats(self):
        """Compute yield standardization statistics."""
        all_yields = []

        for seq in self.sequences:
            # Only use known yields for computing statistics
            known_mask = seq["mask"]
            known_yields = seq["yields"][known_mask]
            all_yields.extend(known_yields)

        all_yields = np.array(all_yields)
        self.yield_stats = {"mean": np.mean(all_yields), "std": np.std(all_yields)}

    def _standardize_yields(self):
        """Standardize yield values."""
        mean = self.yield_stats["mean"]
        std = self.yield_stats["std"]

        for seq in self.sequences:
            seq["yields_std"] = (seq["yields"] - mean) / std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return a sequence sample.
        Model expects: year, coords, period, yield, mask
        """
        seq = self.sequences[idx]
        seq_len = self.total_years

        # Prepare input tensors
        years = torch.tensor(seq["years"], dtype=torch.float32)

        # Use actual lat/lng coordinates for this county
        coords = torch.full((seq_len, 2), fill_value=0.0, dtype=torch.float32)
        coords[:, 0] = seq["lat"]  # latitude
        coords[:, 1] = seq["lng"]  # longitude

        # For periods, pad with 0 for first element
        periods_padded = np.concatenate([[0], seq["periods"]])
        periods = torch.tensor(periods_padded, dtype=torch.float32)

        # Yields (standardized values)
        yields = torch.tensor(seq["yields_std"], dtype=torch.float32)

        # Mask for known vs unknown yields (True = known, False = masked/unknown)
        yield_mask = torch.tensor(seq["mask"], dtype=torch.bool)

        # Create padding mask for transformer (True = padding, False = valid)
        # All positions are valid in our case since we have full sequences
        padding_mask = torch.zeros(seq_len, dtype=torch.bool)

        return years, coords, periods, yields, yield_mask, padding_mask


class SeqDataloader:
    def __init__(
        self,
        crop_type: str,
        batch_size: int,
        test_year_cutoff: int,
        known_years: int,
        total_years: int,
    ):
        """
        Dataloader for crop yield sequence data with masking.

        Args:
            crop_type: Type of crop (e.g., 'corn', 'winter_wheat')
            batch_size: Batch size for dataloaders
            test_year_cutoff: Years > this value go to test set
            known_years: Number of known datapoints in the small history
            total_years: Total sequence length after backward expansion
        """
        self.crop_type = crop_type
        self.data_path = f"data/USDA/{crop_type}_yield_processed.csv"
        self.batch_size = batch_size
        self.test_year_cutoff = test_year_cutoff
        self.known_years = known_years
        self.total_years = total_years

        # Load data
        self.data = pd.read_csv(self.data_path)

        # Create train/test split by year
        self.train_data, self.test_data = self._create_train_test_split()

        # Create datasets once during initialization
        self.train_dataset = SeqDataset(
            self.train_data,
            known_years=self.known_years,
            total_years=self.total_years,
            yield_stats=None,
        )

        # Create test dataset using train yield statistics
        self.test_dataset = SeqDataset(
            self.test_data,
            known_years=self.known_years,
            total_years=self.total_years,
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
