from torch.utils.data import DataLoader
import torch.optim as optim
from src.model.constants import *
import json
from tqdm import tqdm


def save_losses(losses, total_params, plot=True):
    with open(
        DATA_DIR + f"weatherformer_{total_params / 10**6:.1f}m_losses.json", "w"
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


def streaming_dataloader(
    batch_size,
    split="train",
    shuffle=False,
    lr_finder=False,
    num_input_features=None,
    num_output_features=None,
):
    data_loader_dir = DATA_DIR + "nasa_power/pytorch/"

    if TEST_ENV or lr_finder:
        train_indices = DRY_RUN_TRAIN_PART_IDS
        test_indices = TEST_PART_IDS[:1]
    else:
        train_indices = set(range(NUM_DATASET_PARTS)).difference(TEST_PART_IDS)
        test_indices = TEST_PART_IDS

    train_loader_paths = [
        data_loader_dir + f"weather_dataset_{frequency}_{i}.pt"
        for i in train_indices
        for frequency in ["monthly", "weekly", "daily"]
    ]
    test_loader_paths = [
        data_loader_dir + f"weather_dataset_{frequency}_{i}.pt"
        for i in test_indices
        for frequency in ["monthly", "weekly", "daily"]
    ]

    file_paths = train_loader_paths if split.lower() == "train" else test_loader_paths

    dataset = StreamingDataset(
        file_paths,
        lr_finder=lr_finder,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_paths,
        lr_finder=False,
        num_input_features=None,
        num_output_features=None,
    ):

        self.file_paths = file_paths
        self.lr_finder = lr_finder
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

    def __iter__(self):
        for file_path in tqdm(
            self.file_paths,
            "Iterating: ",
            file=TQDM_OUTPUT,
            mininterval=2 * 60,  # seconds
        ):
            data = torch.load(file_path, weights_only=False)
            for item in data:
                if self.lr_finder:
                    # create input and target
                    weather, coords, index = item

                    # Create mask of ones and zeros
                    mask = torch.zeros(weather.shape[-1])
                    # Randomly select self.num_input_features indices to set to 1
                    random_indices = torch.randperm(len(mask))[
                        : self.num_input_features
                    ]
                    mask[random_indices] = 1

                    # Expand mask to match weather shape
                    expanded_mask = mask.expand(weather.shape)

                    # Create masked copy
                    masked_weather = weather * expanded_mask

                    # Yield both masked version as input and weather as target
                    yield (masked_weather, coords, index), weather
                else:
                    yield item
