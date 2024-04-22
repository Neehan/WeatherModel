import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm


def train_test_split(dataset_dir, shuffle=True, test_fraction=0.05):
    """
    A generator function that creates and saves train and test datasets from multiple sources,
    each having multiple TensorDataset files stored on disk, without exceeding memory limits.

    Args:
        shuffle (bool): Whether to shuffle the dataset after combining chunks from each source.
    """
    train_loader_paths = {
        frequency: [dataset_dir + f"_{frequency}_{i}.pth" for i in range(8)]
        for frequency in ["daily", "weekly", "monthly"]
    }

    # Assume each source list in the dictionary has the same number of chunks
    num_chunks = len(list(train_loader_paths.values())[0])
    source_keys = list(train_loader_paths.keys())

    for chunk_index in tqdm(
        range(num_chunks), desc="Processing Chunks", dynamic_ncols=True
    ):
        num_tensors = 3
        combined_data = [[] for _ in range(num_tensors)]

        # Load and combine datasets from all sources
        for source in source_keys:
            file_path = train_loader_paths[source][chunk_index]
            dataset = torch.load(file_path)
            for i in range(num_tensors):
                combined_data[i].append(dataset.tensors[i])

        # Concatenate all features and labels from different sources
        for i in range(num_tensors):
            combined_data[i] = torch.cat(combined_data[i], dim=0)

        # Shuffle datasets if required
        if shuffle:
            indices = torch.randperm(combined_data[0].shape[0])
            for i in range(num_tensors):
                combined_data[i] = combined_data[i][indices]

        total_samples = combined_data[0].shape[
            0
        ]  # total_samples should be based on one of the tensors, here it's assumed they all have the same size in the first dimension
        train_size = int(total_samples * (1 - test_fraction))

        # Split the features and labels into train and test sets
        train_data, test_data = [None] * num_tensors, [None] * num_tensors
        for i in range(num_tensors):
            train_data[i] = combined_data[i][:train_size].clone()
            test_data[i] = combined_data[i][train_size:total_samples].clone()

        # Create TensorDataset objects
        train_dataset = TensorDataset(*train_data)
        test_dataset = TensorDataset(*test_data)

        # Save datasets to disk
        torch.save(
            train_dataset,
            dataset_dir.split("weather_")[0] + f"/train_dataset_chunk_{chunk_index}.pt",
        )
        torch.save(
            test_dataset,
            dataset_dir.split("weather_")[0] + f"/test_dataset_chunk_{chunk_index}.pt",
        )


if __name__ == "__main__":
    DATA_DIR = "data/"
    dataset_dir = DATA_DIR + "nasa_power/processed/weather_dataset"

    train_test_split(dataset_dir, shuffle=True)
