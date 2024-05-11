import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import logging
import math
from .constants import *

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def compute_mae(model, data_loader, n_eval_weeks, plot=False):
    model.eval()
    device = DEVICE
    # Compute the mae on the training dataset
    mae_total = 0.0
    outputs = []
    targets = []

    for data in data_loader:
        (
            weather,
            mask,
            weather_index,
            coords,
            ili_past,
            tot_cases_past,
            ili_target,
        ) = (x.to(device) for x in data)

        # Forward pass
        output = model(weather, mask, weather_index, coords, ili_past, tot_cases_past)
        outputs += (
            output.detach()
            .cpu()[:, -1]
            .reshape(
                -1,
            )
            .tolist()
        )
        targets += (
            ili_target.detach()
            .cpu()[:, -1]
            .reshape(
                -1,
            )
            .tolist()
        )
        # Compute the mean squared error
        # if ili_target.shape[0] < batch_size: # last batch
        #     n_eval_weeks =
        mae = F.l1_loss(output[:, :n_eval_weeks], ili_target[:, :n_eval_weeks])

        # Accumulate the MSE over all batches
        mae_total += mae.item()

    if plot:
        plt.plot(range(len(outputs)), outputs, label="outputs")
        plt.plot(range(len(targets)), targets, label="targets")
        plt.legend()
        plt.show()

    return mae_total / len(data_loader)


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
    n_eval_weeks = [1, 5, 10]
    best_test_maes = [999, 999, 999]

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda(
            num_warmup_epochs=num_warmup_epochs, decay_factor=lr_decay_factor
        ),
    )
    # test_mae = compute_mae(model, test_loader, n_eval_weeks, plot=False)
    # print(f"test mae: {test_mae}")
    # return {}, test_mae

    for epoch in tqdm(
        range(num_epochs),
        file=TQDM_OUTPUT,
        desc="Training",
        dynamic_ncols=True,
        mininterval=MIN_INTERVAL,
    ):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(
            train_loader,
        ):
            # Zero the gradients
            optimizer.zero_grad()
            (
                weather,
                mask,
                weather_index,
                coords,
                ili_past,
                tot_cases_past,
                ili_target,
            ) = (x.to(device) for x in data)

            # Forward pass
            outputs = model(
                weather, mask, weather_index, coords, ili_past, tot_cases_past
            )
            loss = criterion(outputs, ili_target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # logging.info statistics
            running_loss += loss.item()
        scheduler.step()
        running_loss /= len(train_loader)
        # running_loss = math.sqrt(running_loss)
        losses["train"].append(running_loss)
        # losses["test"].append(test_mae)
        logging_text = f"[{epoch+1} / {num_epochs}] Test MAE best:"
        for i, n_eval_week in enumerate(n_eval_weeks):
            test_mae = compute_mae(model, test_loader, n_eval_week)
            best_test_maes[i] = min(test_mae, best_test_maes[i])
            logging_text += f" {best_test_maes[i]:.3f};"
        if epoch % 5 == 4 or epoch == num_epochs - 1:
            logging.info(logging_text)
        # logging.info(f"[{epoch+1} / {num_epochs}] Loss: {running_loss:3f}")
    return losses, best_test_maes
