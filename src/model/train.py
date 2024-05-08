from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from . import utils
from .constants import *


def swap_feature(weather_indices, num_input_features):
    # Select one index randomly from the first input_dim indices
    input_index = torch.randint(0, num_input_features, (1,))
    # Select one index from the remaining indices
    output_index = torch.randint(num_input_features, len(weather_indices), (1,))
    # Swap them
    weather_indices[input_index], weather_indices[output_index] = (
        weather_indices[output_index],
        weather_indices[input_index],
    )
    return weather_indices


def train(
    model,
    num_input_features,
    loader,
    weather_indices,
    optimizer,
    scheduler,
    criterion,
    device,
):
    model.train()
    total_loss = 0
    loader_len = 0
    for (
        data,
        coords,
        index,
    ) in loader:
        data, coords, index = data.to(device), coords.to(device), index.to(device)
        optimizer.zero_grad()

        # swap one input and output feature of weather indices
        swap_feature(weather_indices, num_input_features)
        target_indices = weather_indices[num_input_features:]
        target_features = data[:, :, target_indices]
        target_mask = torch.zeros(TOTAL_WEATHER_VARS, dtype=torch.bool, device=DEVICE)
        target_mask[target_indices] = True

        output = model(data, coords, index, weather_feature_mask=target_mask)[
            :, :, target_indices
        ]

        loss = criterion(target_features, output)

        total_loss += loss.item()
        loader_len += 1

        # Backward pass
        loss.backward()
        optimizer.step()
    scheduler.step()

    return np.sqrt(total_loss / loader_len)


@torch.no_grad()
def validate(
    model,
    num_input_features,
    loader,
    weather_indices,
    device,
):
    model.eval()
    total_loss = 0
    loader_len = 0
    for (
        data,
        coords,
        index,
    ) in loader:
        data, coords, index = data.to(device), coords.to(device), index.to(device)
        # swap one input and output feature of weather indices
        swap_feature(weather_indices, num_input_features)
        target_indices = weather_indices[num_input_features:]
        target_features = data[:, :, target_indices]
        target_mask = torch.zeros(TOTAL_WEATHER_VARS, dtype=torch.bool, device=DEVICE)
        target_mask[target_indices] = True

        output = model(data, coords, index, weather_feature_mask=target_mask)[
            :, :, target_indices
        ]

        loss = F.mse_loss(target_features, output)

        total_loss += loss.item()
        loader_len += 1

    return np.sqrt(total_loss / loader_len)


def training_loop(
    model,
    batch_size,
    num_input_features,
    num_output_features,
    num_epochs,
    init_lr,
    num_warmup_epochs,
    decay_factor,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params/10**6:.2f}M")

    feature_dim = num_input_features + num_output_features
    weather_indices = torch.arange(feature_dim)

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = utils.get_scheduler(optimizer, num_warmup_epochs, decay_factor)
    criterion = nn.MSELoss()

    losses = {
        "train": [],
        "val": [],
    }

    data_loader_dir = DATA_DIR + "nasa_power/processed/weather_dataset"

    train_indices = set(range(NUM_DATASET_PARTS)).difference(TEST_PART_IDS)
    train_loader_paths = [data_loader_dir + f"_combined_{i}.pt" for i in train_indices]
    test_loader_paths = [data_loader_dir + f"_combined_{i}.pt" for i in TEST_PART_IDS]

    for epoch in range(num_epochs):
        train_loader = utils.streaming_dataloader(
            train_loader_paths, batch_size, shuffle=True, split="train"
        )

        test_loader = utils.streaming_dataloader(
            test_loader_paths, batch_size, shuffle=True, split="validation"
        )

        train_loss = train(
            model,
            num_input_features,
            train_loader,
            weather_indices,
            optimizer,
            scheduler,
            criterion,
            DEVICE,
        )

        val_loss = validate(
            model,
            num_input_features,
            test_loader,
            weather_indices,
            DEVICE,
        )

        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        daily_scaler_mean = model.input_scaler.weight[1].mean().item()
        weekly_scaler_mean = model.input_scaler.weight[7].mean().item()
        logging.info(
            f"Epoch {epoch+1}: Losses train: {train_loss:.3f} val: {val_loss:.3f}, scaler means: daily {daily_scaler_mean:.3f}, weekly: {weekly_scaler_mean:.3f}"
        )
        if epoch % 2 == 1 or epoch == num_epochs - 1:
            torch.save(
                model,
                DATA_DIR
                + f"trained_models/weatherformer_{total_params / 10**6:.1f}m_epoch_{epoch}.pth",
            )
            torch.save(
                model,
                DATA_DIR
                + f"trained_models/weatherformer_{total_params / 10**6:.1f}m_latest.pth",
            )
            utils.save_losses(losses, total_params)
    return model, losses
