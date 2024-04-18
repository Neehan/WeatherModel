import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import logging
import math
from .constants import *


def compute_rmse(model, data_loader):
    model.eval()
    device = DEVICE
    # Compute the RMSE on the training dataset
    mse_total = 0.0
    for weather, practices, soil, year, coord, y, y_past, mask in data_loader:
        weather = weather.to(device)
        soil = soil.to(device)
        year = year.to(device)
        practices = practices.to(device)
        coord = coord.to(device)
        y = y.to(device)
        y_past = y_past.to(device)
        mask = mask.to(device)

        # Forward pass
        outputs = model(weather, soil, practices, year, coord, y_past, mask)
        # Compute the mean squared error
        mse = F.mse_loss(outputs, y.to(device))

        # Accumulate the MSE over all batches
        mse_total += mse.item()

    # Compute the RMSE
    rmse = math.sqrt(mse_total / len(data_loader))
    return rmse


# Warm-up and decay function
def lr_lambda(num_warmup_epochs=10, decay_factor=0.95):
    def fn(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warm-up
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Exponential decay
            return decay_factor ** (current_epoch - num_warmup_epochs)

    return fn


def training_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs=20,
    init_lr=0.0009,
    num_warmup_epochs=2,
    lr_decay_factor=0.95,
):

    losses = {
        "train": [],
        "test": [],
    }

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params}")
    device = DEVICE

    criterion = nn.MSELoss()
    best_test_rmse = 999

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda(
            num_warmup_epochs=num_warmup_epochs, decay_factor=lr_decay_factor
        ),
    )

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (
            weather,
            practices,
            soil,
            year,
            coord,
            y,
            y_past,
            mask,
        ) in enumerate(
            tqdm(
                train_loader,
                file=TQDM_OUTPUT,
                desc="Training",
                dynamic_ncols=True,
                mininterval=MIN_INTERVAL,
            )
        ):
            # Zero the gradients
            optimizer.zero_grad()
            weather = weather.to(device)
            soil = soil.to(device)
            year = year.to(device)
            practices = practices.to(device)
            coord = coord.to(device)
            y = y.to(device)
            y_past = y_past.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(weather, soil, practices, year, coord, y_past, mask)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # logging.info statistics
            running_loss += loss.item()
        scheduler.step()
        running_loss /= len(train_loader)
        running_loss = math.sqrt(running_loss)
        losses["train"].append(running_loss)
        test_rmse = compute_rmse(model, test_loader)
        losses["test"].append(test_rmse)
        best_test_rmse = min(test_rmse, best_test_rmse)
        logging.info(
            f"[{epoch+1} / {num_epochs} Test RMSE best: {best_test_rmse:.3f}, current: {test_rmse:.3f}"
        )
        logging.info(f"[{epoch+1} / {num_epochs}] Loss: {running_loss:3f}")
    return model, losses
