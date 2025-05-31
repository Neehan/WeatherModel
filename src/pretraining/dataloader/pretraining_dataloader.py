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

random.seed(1234)
logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    """
    Initialize each worker process with a unique random seed.
    This ensures reproducible but different random states across workers.
    """
    # Get worker info to determine the unique seed
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Create unique seed for this worker: base_seed + worker_id + dataset_rank
        base_seed = 1234
        dataset_rank = getattr(worker_info.dataset, "rank", 0)  # Safe access to rank
        unique_seed = base_seed + worker_id + (dataset_rank * 1000)

        # Set seeds for this worker
        random.seed(unique_seed)
        torch.manual_seed(unique_seed)

        # Log worker initialization (only occasionally to avoid spam)
        if worker_id == 0:
            logger.debug(f"Worker {worker_id} initialized with seed {unique_seed}")


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths,
        num_input_features=None,
        num_output_features=None,
        shuffle=False,
        masking_function: Optional[str] = None,
        masking_prob: float = 0.15,
        n_masked_features: int = 1,
        rank: int = 0,
    ):

        self.file_paths = file_paths
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.shuffle = shuffle
        self.masking_prob = masking_prob
        self.n_masked_features = n_masked_features
        self.rank = rank

        # logger.info(
        #     f"Masking function: {masking_function}, masking prob: {masking_prob}, n_masked_features: {n_masked_features}"
        # )
        if masking_function == "weatherbert":
            self.masking_function = self.weatherbert_masking_function
        elif masking_function == "weatherformer":
            self.masking_function = self.weatherformer_masking_function
        else:
            raise ValueError(f"Masking function {masking_function} is not valid")

    def weatherbert_masking_function(self, seq_len, n_features):
        """
        BERT-style masking for weather data.
        Randomly masks features across the sequence based on masking_prob.
        """
        # Generate random probabilities for all positions at once
        random_probs = torch.rand(seq_len, n_features)

        # Create mask where random probability is less than masking probability
        mask = random_probs < self.masking_prob

        return mask

    def weatherformer_masking_function(self, seq_len, n_features):
        """
        WeatherFormer-style masking for weather data.
        Masks n_masked_features completely across all timesteps.
        """
        # Create mask initialized to False (no masking)
        mask = torch.zeros(seq_len, n_features, dtype=torch.bool)

        # Randomly select n_masked_features to mask completely
        masked_feature_indices = torch.randperm(n_features)[: self.n_masked_features]

        # Mask the selected features across all timesteps
        mask[:, masked_feature_indices] = True

        return mask

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

                    # Calculate years for each of the 365 time points
                    # temporal_index is the segment number (0, 1, 2, ...)
                    # Each segment has 365 time points
                    # absolute_time_index = segment * 365 + position_in_segment
                    seq_len = weather.size(0)  # Should be 365
                    time_point_indices = torch.arange(seq_len, dtype=torch.float32)
                    absolute_time_indices = temporal_index * 365 + time_point_indices

                    # Convert to years: 1984 + (absolute_time_index * interval_days) / 365
                    years = 1984 + (absolute_time_indices * interval) / 365

                    # Apply masking function if provided
                    if self.masking_function is not None:
                        seq_len, n_features = weather.size()
                        feature_mask = self.masking_function(seq_len, n_features)
                        all_samples.append(
                            (weather, coords, years, interval, feature_mask)
                        )
                    else:
                        all_samples.append((weather, coords, years, interval))

            # Shuffle all samples from this chunk if shuffle is enabled
            if self.shuffle:
                random.shuffle(all_samples)

            # Yield all samples
            for sample in all_samples:
                # print(f"year: {sample[2]}, interval: {sample[3]}, coords: {sample[1]}")
                yield sample

            # Only log from rank 0 and only every 5 chunks to reduce spam
            if (i // 3) % 10 == 0 and self.rank == 0:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None or worker_info.id == 0:
                    logger.info(
                        f"Dataloader iterated over [{(i//3)+1}/{len(self.file_paths)//3}] chunks"
                    )


def streaming_dataloader(
    batch_size,
    split="train",
    shuffle=False,
    num_input_features=None,
    num_output_features=None,
    masking_function: Optional[str] = None,
    masking_prob: float = 0.15,
    n_masked_features: int = 1,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,  # Add num_workers parameter
):
    data_loader_dir = DATA_DIR + "nasa_power/pytorch/"

    if DRY_RUN:
        train_indices = DRY_RUN_TRAIN_CHUNK_IDS
        test_indices = VALIDATION_CHUNK_IDS[:4]  # Use 4 validation chunks for 4 GPUs
    else:
        train_indices = set(range(NUM_DATASET_PARTS)).difference(VALIDATION_CHUNK_IDS)
        test_indices = VALIDATION_CHUNK_IDS

    # Convert to lists and create different orderings for each frequency
    indices = train_indices if split.lower() == "train" else test_indices
    indices_list = list(indices)

    # For distributed training, partition the data across ranks
    if world_size > 1:
        # Ensure all GPUs get exactly the same number of chunks
        # Truncate to largest number evenly divisible by world_size
        num_chunks_per_gpu = len(indices_list) // world_size
        total_chunks_to_use = num_chunks_per_gpu * world_size
        indices_list = indices_list[:total_chunks_to_use]

        # Distribute indices across ranks
        indices_per_rank = len(indices_list) // world_size
        start_idx = rank * indices_per_rank
        end_idx = start_idx + indices_per_rank
        indices_list = indices_list[start_idx:end_idx]

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
        masking_function=masking_function,
        masking_prob=masking_prob,
        n_masked_features=n_masked_features,
        rank=rank,
    )

    # Determine optimal number of workers
    # For distributed training, reduce workers per GPU to avoid resource contention
    effective_num_workers = num_workers
    if world_size > 1:
        # Reduce workers in distributed setting to avoid overwhelming the system
        effective_num_workers = min(num_workers, 3)

    # Log worker configuration
    if rank == 0:
        logger.info(
            f"Using {effective_num_workers} workers per GPU (world_size={world_size})"
        )

    effective_num_workers = 0
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=effective_num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=(
            True if effective_num_workers > 0 else False
        ),  # Keep workers alive between epochs
    )
