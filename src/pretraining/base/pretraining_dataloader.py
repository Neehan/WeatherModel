import torch
import torch.utils.data
from torch.utils.data import DataLoader
import logging
import random
from typing import Callable, Optional
from src.utils.constants import (
    DATA_DIR,
    DRY_RUN,
    DRY_RUN_TRAIN_CHUNK_IDS,
    VALIDATION_CHUNK_IDS,
    NUM_DATASET_PARTS,
)
from src.utils.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths,
        num_input_features=None,
        num_output_features=None,
        shuffle=False,
    ):

        self.file_paths = file_paths
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.shuffle = shuffle

    def __iter__(self):
        # Process files in groups of 3 (monthly, weekly, daily with different indices)
        for i in range(0, len(self.file_paths), 3):
            chunk_files = self.file_paths[i : i + 3]  # monthly_i, weekly_j, daily_k

            # Load all three frequency files
            frequency_data = []
            for file_path in chunk_files:
                data = torch.load(file_path, weights_only=False)
                frequency_data.append(data)

            # Collect all samples from the three frequency files
            all_samples = []
            for data in frequency_data:
                for weather, coords, index in data:
                    # index is (temporal index since 1984 jan 1, temporal interval)
                    temporal_index = index[:1]
                    interval = index[1:]

                    # Convert temporal index to year
                    year = (
                        1984 + (temporal_index * interval.int() // 365).int()
                    ).float()

                    all_samples.append((weather, coords, year, interval))

            # Shuffle all samples from this chunk if shuffle is enabled
            if self.shuffle:
                random.shuffle(all_samples)

            # Yield all samples
            for sample in all_samples:
                # print(f"year: {sample[2]}, interval: {sample[3]}, coords: {sample[1]}")
                yield sample

            if (i // 3) % 5 == 0:
                logger.info(
                    f"Dataloader iterated over [{(i//3)+1}/{len(self.file_paths)//3}] chunks"
                )


class MaskingCollate:
    """Pickleable collate function that applies masking after batching."""

    def __init__(self, masking_function: Optional[Callable] = None):
        self.masking_function = masking_function

    def __call__(self, batch):
        # Separate the elements
        weather_list, coords_list, year_list, interval_list = zip(*batch)

        # Stack into batches
        weather = torch.stack(weather_list)
        coords = torch.stack(coords_list)
        year = torch.stack(year_list)
        interval = torch.stack(interval_list)

        # Apply masking function if provided
        if self.masking_function is not None:
            batch_size, seq_len, n_features = weather.size()
            feature_mask = self.masking_function(batch_size, seq_len, n_features)
            return weather, coords, year, interval, feature_mask
        else:
            return weather, coords, year, interval


def streaming_dataloader(
    batch_size,
    split="train",
    shuffle=False,
    lr_finder=False,
    num_input_features=None,
    num_output_features=None,
    masking_function: Optional[Callable] = None,
):
    data_loader_dir = DATA_DIR + "nasa_power/pytorch/"

    if DRY_RUN or lr_finder:
        train_indices = DRY_RUN_TRAIN_CHUNK_IDS
        test_indices = VALIDATION_CHUNK_IDS[:1]
    else:
        train_indices = set(range(NUM_DATASET_PARTS)).difference(VALIDATION_CHUNK_IDS)
        test_indices = VALIDATION_CHUNK_IDS

    # Convert to lists and create different orderings for each frequency
    indices = train_indices if split.lower() == "train" else test_indices
    indices_list = list(indices)

    # Create different shuffled orderings for each frequency
    monthly_indices = indices_list.copy()
    weekly_indices = indices_list.copy()
    daily_indices = indices_list.copy()

    if shuffle:
        random.shuffle(weekly_indices)
        random.shuffle(daily_indices)

    # Create file paths with rearranged indices
    file_paths = []
    for i in range(len(indices_list)):
        monthly_idx = monthly_indices[i]
        weekly_idx = weekly_indices[i]
        daily_idx = daily_indices[i]

        file_paths.extend(
            [
                data_loader_dir + f"weather_dataset_monthly_{monthly_idx}.pt",
                data_loader_dir + f"weather_dataset_weekly_{weekly_idx}.pt",
                data_loader_dir + f"weather_dataset_daily_{daily_idx}.pt",
            ]
        )

    dataset = StreamingDataset(
        file_paths,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        shuffle=shuffle,
    )

    # Create pickleable collate function with masking
    collate_fn = MaskingCollate(masking_function)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
