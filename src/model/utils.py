from torch.utils.data import DataLoader
import torch.optim as optim
from src.model.constants import *
import json
from tqdm import tqdm
from torch_lr_finder import LRFinder


def save_losses(losses, total_params, plot=True):
    with open(
        DATA_DIR + f"weatherformer_{total_params / 10**6:.1f}m_losses.json", "w"
    ) as f:
        json.dump(losses, f)
    f.close()


def find_optimal_lr(model, criterion, optimizer, train_loader, device):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

    # Get the optimal learning rate
    optimal_lr = lr_finder.suggested_lr

    # Reset the model and optimizer to their initial state
    lr_finder.reset()

    return optimal_lr


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


def streaming_dataloader(
    file_paths,
    batch_size,
    split="train",
    shuffle=False,
    lr_finder=False,
    num_input_features=None,
    num_output_features=None,
):
    dataset = StreamingDataset(
        file_paths,
        split=split,
        lr_finder=lr_finder,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths,
        split="train",
        lr_finder=False,
        num_input_features=None,
        num_output_features=None,
    ):
        self.file_paths = file_paths
        self.split = split
        self.lr_finder = lr_finder
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

    def __iter__(self):
        for file_path in self.file_paths:
            data = torch.load(file_path)
            for item in data[self.split]:
                # finding the optimal lr, so output in a specific way
                if self.lr_finder:
                    # Create mask of ones and zeros
                    mask = torch.zeros(
                        self.num_input_features + self.num_output_features
                    )
                    mask[: self.num_input_features] = 1

                    # Expand mask to match item shape
                    expanded_mask = mask.expand(item.shape[:-1] + (-1,))

                    # Create masked copy
                    masked_item = item * expanded_mask

                    # Yield both original and masked versions
                    yield masked_item, item
                else:
                    yield item
