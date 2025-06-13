import torch
import torch.utils.data
import logging
import random
from typing import Optional
from src.utils.constants import (
    DATA_DIR,
    DRY_RUN,
    DRY_RUN_TRAIN_CHUNK_IDS,
    VALIDATION_CHUNK_IDS,
    NUM_DATASET_PARTS,
)
import logging

random.seed(1234)
logger = logging.getLogger(__name__)


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

        # Set the correct device for this rank
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        # logger.info(
        #     f"Masking function: {masking_function}, masking prob: {masking_prob}, n_masked_features: {n_masked_features}"
        # )
        if masking_function == "weatherbert":
            self.masking_function = self.weatherbert_masking_function
        elif masking_function == "weatherformer":
            self.masking_function = self.weatherformer_masking_function
        else:
            raise ValueError(f"Masking function {masking_function} is not valid")

    def weatherbert_masking_function(self, seq_len, n_features, batch_size):
        """
        BERT-style masking for weather data.
        Randomly masks features across the sequence based on masking_prob.
        """
        # Generate random probabilities for all positions at once on GPU
        random_probs = torch.rand(batch_size, seq_len, n_features, device=self.device)

        # Create mask where random probability is less than masking probability
        mask = random_probs < self.masking_prob
        return mask

    def weatherformer_masking_function(self, seq_len, n_features, batch_size):
        """
        WeatherFormer-style masking for weather data.
        Masks n_masked_features completely across all timesteps.
        """
        # Generate random values and sort to get permutations: [batch_size, n_features]
        random_vals = torch.rand(batch_size, n_features, device=self.device)
        rand_perm = torch.argsort(random_vals, dim=-1)

        # Create feature mask: True where perm value < n_masked_features
        # This selects exactly n_masked_features random features per sample
        feature_mask = rand_perm < self.n_masked_features  # [batch_size, n_features]

        # Expand to all timesteps: [batch_size, seq_len, n_features]
        mask = feature_mask.unsqueeze(1).expand(-1, seq_len, -1)

        return mask

    def __iter__(self):
        """
        returns:
            weather: [batch_size, seq_len, n_features]
            coords: [batch_size, 2]
            year: [batch_size, seq_len]
            interval: [batch_size]
            mask: [batch_size, seq_len, n_features]
        """
        # Process files in groups of 3 (monthly, weekly, daily with different indices)
        for i in range(0, len(self.file_paths), 3):
            chunk_files = self.file_paths[i : i + 3]  # monthly_i, weekly_j, daily_k
            # chunk_files = self.file_paths[i + 1 : i + 2]  # weekly_j,

            # Load all three frequency files directly to GPU
            all_data = []
            for file_path in chunk_files:
                data = torch.load(
                    file_path, weights_only=False, map_location=self.device
                )
                all_data.extend(data)

            if not all_data:
                continue

            # Pre-calculate total number of samples for efficient tensor allocation
            total_samples = len(all_data)

            # Get dimensions from first sample
            first_weather, first_coords, _ = all_data[0]
            seq_len, n_features = (
                first_weather.shape
            )  # seq_len=365, n_features=weather_vars
            coord_dim = first_coords.shape[0]  # coord_dim=2 (lat, lon)

            # Pre-allocate tensors for all samples (much more efficient than appending)
            weather_tensors = torch.zeros(
                total_samples, seq_len, n_features, device=self.device
            )
            coords_tensors = torch.zeros(total_samples, coord_dim, device=self.device)
            years_tensors = torch.zeros(total_samples, seq_len, device=self.device)
            interval_tensors = torch.zeros(total_samples, 1, device=self.device)

            if self.masking_function is not None:
                mask_tensors = torch.zeros(
                    total_samples,
                    seq_len,
                    n_features,
                    dtype=torch.bool,
                    device=self.device,
                )

            # Vectorized data processing - much faster than individual loops
            for idx, (weather, coords, index) in enumerate(all_data):
                temporal_index = index[0]  # Scalar: temporal segment index
                interval = index[1]  # Scalar: days per timestep (1, 7, or 30)

                # Store data directly in pre-allocated tensors
                weather_tensors[idx] = weather  # [seq_len, n_features] -> [365, 20]
                coords_tensors[idx] = coords  # [coord_dim] -> [2]
                interval_tensors[idx, 0] = interval  # scalar -> assign to [idx, 0]

                # Vectorized year calculation for entire sequence at once
                # Shape: [seq_len] - e.g., [365] for time indices 0, 1, 2, ..., 364
                time_point_indices = torch.arange(
                    seq_len, dtype=torch.float32, device=self.device
                )
                # Shape: [seq_len] - absolute time indices since 1984
                absolute_time_indices = temporal_index * 365 + time_point_indices
                # Shape: [seq_len] - convert to actual years
                years_tensors[idx] = 1984.0 + (absolute_time_indices * interval) / 365

            if self.masking_function is not None:
                # Use the actual masking functions with batch support
                mask_tensors = self.masking_function(seq_len, n_features, total_samples)

            # GPU-based shuffling using direct tensor indexing (much faster)
            if self.shuffle and total_samples > 1:
                shuffle_indices = torch.randperm(total_samples, device=self.device)
                weather_tensors = weather_tensors[shuffle_indices]
                coords_tensors = coords_tensors[shuffle_indices]
                years_tensors = years_tensors[shuffle_indices]
                interval_tensors = interval_tensors[shuffle_indices]

                if self.masking_function is not None:
                    mask_tensors = mask_tensors[shuffle_indices]

            # Yield all samples (vectorized unpacking)
            for sample_idx in range(total_samples):
                if self.masking_function is not None:
                    sample = (
                        weather_tensors[sample_idx],
                        coords_tensors[sample_idx],
                        years_tensors[sample_idx],
                        interval_tensors[sample_idx],
                        mask_tensors[sample_idx],
                    )
                else:
                    sample = (
                        weather_tensors[sample_idx],
                        coords_tensors[sample_idx],
                        years_tensors[sample_idx],
                        interval_tensors[sample_idx],
                    )
                yield sample

            # Only log from rank 0 and only every 10 chunks to reduce spam
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,  # No need for pin_memory since data is already on GPU
        num_workers=0,
    )
