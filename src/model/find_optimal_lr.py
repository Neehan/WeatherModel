from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch_lr_finder import LRFinder


from src.model import utils
from src.model.constants import *


def find_optimal_lr(
    model,
    batch_size,
    num_input_features,
    num_output_features,
    init_lr,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params/10**6:.2f}M")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-2)

    # Find the optimal learning rate
    train_loader = utils.streaming_dataloader(
        batch_size,
        shuffle=True,
        split="train",
        lr_finder=True,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
    )
    lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    ax, optimal_lr = lr_finder.plot(suggest_lr=True)
    # Reset the model and optimizer to their initial state
    lr_finder.reset()
    logging.info(f"optimal learning rate: {optimal_lr:.6f}")

    return optimal_lr
