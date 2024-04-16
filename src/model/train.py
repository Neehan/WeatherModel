from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import utils
from constants import *


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

        output = model(data, coords, index, weather_feature_mask=target_indices)[
            :, :, target_indices
        ]

        loss = criterion(target_features, output)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
    scheduler.step()

    return np.sqrt(total_loss / len(loader))


def training_loop(
    model,
    train_loader,
    num_input_features,
    num_output_features,
    num_epochs,
    init_lr,
    num_warmup_epochs,
    decay_factor,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params}")

    feature_dim = num_input_features + num_output_features
    weather_indices = torch.arange(feature_dim)

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = utils.get_scheduler(optimizer, num_warmup_epochs, decay_factor)
    criterion = nn.MSELoss()

    losses = {
        "train": [],
    }

    for epoch in range(num_epochs):
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

        losses["train"].append(train_loss)

        mask_value = model.mask_value.item()
        input_scaler = model.input_scaler.item()
        logging.info(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, mask: {mask_value:.4f}, scaler: {input_scaler:.4f}"
        )
        if epoch % 5 == 4 or epoch == num_epochs - 1:
            torch.save(
                model,
                DATA_DIR
                + f"weatherformer_{total_params / 10**6:.1f}m_epoch_{epoch}.pth",
            )
            utils.save_losses(losses, total_params)
    return model, losses