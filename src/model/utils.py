from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from constants import *
import json
from tqdm import tqdm


def save_losses(losses, total_params):
    with open(
        DATA_DIR + f"weatherformer_{total_params / 10**6:.1f}_losses.json", "w"
    ) as f:
        json.dump(losses, f)
    f.close()


def get_scheduler(optimizer, num_warmup_epochs, decay_factor):
    # Warm-up and decay function for scheduler
    def _lr_lambda(num_warmup_epochs=15, decay_factor=0.99):
        def lr_function(current_epoch):
            if current_epoch < num_warmup_epochs:
                return float(current_epoch) / float(max(1, num_warmup_epochs))
            else:
                return decay_factor ** (current_epoch - num_warmup_epochs)

        return lr_function

    return optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(num_warmup_epochs, decay_factor)
    )


def streaming_dataloader(file_paths_dict, batch_size=32, shuffle=True):
    """
    A generator function that streams data from multiple sources,
    each having multiple TensorDataset files stored on disk.

    Args:
    file_paths_dict (dict): A dictionary where keys are source identifiers and values are lists of file paths.
    batch_size (int): The size of each batch.
    shuffle (bool): Whether to shuffle the dataset after combining chunks from each source.

    Yields:
    Tensor: Yields one batch of data at a time as specified by the DataLoader.
    """
    # Assume each source list in the dictionary has the same number of chunks
    num_chunks = len(list(file_paths_dict.values())[0])
    source_keys = list(file_paths_dict.keys())

    for chunk_index in tqdm(
        range(num_chunks), desc="Training", file=TQDM_OUTPUT, dynamic_ncols=True
    ):
        combined_dataset = []

        for source in source_keys:
            file_path = file_paths_dict[source][chunk_index]
            dataset = torch.load(file_path)  # Ensure that this returns a TensorDataset
            combined_dataset.append(dataset)

        if len(combined_dataset) > 1:
            combined_dataset = ConcatDataset(combined_dataset)
        else:
            combined_dataset = combined_dataset[0]

        dataloader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=shuffle
        )

        # Yield batches from the combined DataLoader
        for data in dataloader:
            yield data