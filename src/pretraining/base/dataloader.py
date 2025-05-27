import torch
import torch.utils.data
from torch.utils.data import DataLoader
import logging
import random
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

            if self.shuffle:
                random.shuffle(frequency_data)

            for data in frequency_data:
                for weather, coords, index in data:
                    coords = coords / 360.0
                    index[1] = index[1] / 30.0
                    yield (weather, coords, index)
            if (i // 3) % 5 == 0:
                logger.info(
                    f"Dataloader iterated over [{(i//3)+1}/{len(self.file_paths)//3}] chunks"
                )


def streaming_dataloader(
    batch_size,
    split="train",
    shuffle=False,
    lr_finder=False,
    num_input_features=None,
    num_output_features=None,
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
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
