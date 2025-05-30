import torch.optim as optim
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
import logging
from argparse import ArgumentParser


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


def normalize_year_interval_coords(year, interval, coords):
    year = (year - 1970) / 100.0
    interval = interval / 30.0
    # Create a copy to avoid in-place modification
    coords = coords.clone()
    coords[:, 0] = coords[:, 0] / 360.0
    coords[:, 1] = coords[:, 1] / 180.0
    return year, interval, coords


def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        # Set device for this process
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU training
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Configure logging only for rank 0
def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def get_model_params(model_size: str):
    if model_size == "mini":
        model_size_params = {"num_heads": 4, "num_layers": 2, "hidden_dim_factor": 12}
    elif model_size == "small":
        model_size_params = {"num_heads": 10, "num_layers": 4, "hidden_dim_factor": 20}
    elif model_size == "medium":
        model_size_params = {"num_heads": 12, "num_layers": 6, "hidden_dim_factor": 28}
    elif model_size == "large":
        model_size_params = {"num_heads": 16, "num_layers": 8, "hidden_dim_factor": 36}
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    return model_size_params


def parse_args(parser: ArgumentParser) -> dict:
    args = parser.parse_args()
    args_dict = vars(args)

    logger = logging.getLogger(__name__)
    logger.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logger.info(f"{arg}: {value}")

    # Model size configuration
    model_size = args.model_size.lower()
    model_size_params = get_model_params(model_size)

    args_dict["model_size_params"] = model_size_params

    return args_dict
