from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from src.model import utils
from src.model.constants import *


def swap_features(weather_indices, num_input_features, k=1):
    """
    swap k indices between input and output
    """
    num_output_indices = len(weather_indices) - num_input_features
    k = min(k, num_input_features, num_output_indices)

    # Generate 'k' random indices from the first part of the array
    input_indices = torch.randperm(num_input_features)[:k]

    # Generate 'k' random indices from the second part of the array
    output_indices = num_input_features + torch.randperm(num_output_indices)[:k]

    # Perform the swaps
    weather_indices[input_indices], weather_indices[output_indices] = (
        weather_indices[output_indices].clone(),
        weather_indices[input_indices].clone(),
    )
    return weather_indices


def train(
    model,
    num_input_features,
    loader,
    weather_indices,
    num_feature_swaps,
    optimizer,
    scheduler,
    criterion,
    device,
    enable_gradient_clipping: bool,
):
    model.train()
    total_loss = 0
    loader_len = 0
    for (
        weather,
        coords,
        index,
    ) in loader:
        weather, coords, index = weather.to(device), coords.to(device), index.to(device)
        optimizer.zero_grad()

        # swap some input and output feature of weather indices in place
        swap_features(weather_indices, num_input_features, k=num_feature_swaps)
        target_indices = weather_indices[num_input_features:]
        target_features = weather[:, :, target_indices]
        target_mask = torch.zeros(TOTAL_WEATHER_VARS, dtype=torch.bool, device=DEVICE)
        target_mask[target_indices] = True
        target_mask = target_mask.view(1, -1).expand(weather.shape[0], -1)

        z_mu, z_log_var = model(
            (weather, coords, index),
            weather_feature_mask=target_mask,
            return_log_var=True,
        )

        z_mu, z_log_var = (
            z_mu[:, :, target_indices],
            z_log_var[:, :, target_indices],
        )  # batch_size x seq_len x num_target_indices
        z_std = torch.exp(
            z_log_var / 2
        )  # variance of the latent variable z shape batch_size x seq_len x num_target_indices

        reconstruction_loss = criterion(target_features / z_std, z_mu / z_std)
        variance = z_log_var.mean()

        logging.info(f"reconstruction loss: {reconstruction_loss.item():.4f}")
        logging.info(f"variance: {variance.item():.4f}")

        loss = reconstruction_loss + variance

        total_loss += loss.item()
        loader_len += 1

        # Backward pass
        loss.backward()

        if enable_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
    scheduler.step()

    return total_loss / loader_len


def validate(
    model,
    num_input_features,
    loader,
    weather_indices,
    num_feature_swaps,
    device,
):
    model.eval()
    total_loss = 0
    loader_len = 0
    weather_indices = weather_indices.clone()  # don't modify the input weather indices
    for (
        weather,
        coords,
        index,
    ) in loader:
        weather, coords, index = weather.to(device), coords.to(device), index.to(device)
        # swap one input and output feature of weather indices in place
        swap_features(weather_indices, num_input_features, k=num_feature_swaps)
        target_indices = weather_indices[num_input_features:]
        target_features = weather[:, :, target_indices]
        target_mask = torch.zeros(TOTAL_WEATHER_VARS, dtype=torch.bool, device=DEVICE)
        target_mask[target_indices] = True
        target_mask = target_mask.view(1, -1).expand(weather.shape[0], -1)

        with torch.no_grad():
            z_mu, z_log_var = model(
                (weather, coords, index),
                weather_feature_mask=target_mask,
                return_log_var=True,
            )

        z_mu, z_log_var = (
            z_mu[:, :, target_indices],
            z_log_var[:, :, target_indices],
        )  # batch_size x seq_len x num_target_indices
        z_std = torch.exp(
            z_log_var / 2
        )  # variance of the latent variable z shape batch_size x seq_len x 1

        loss = F.mse_loss(target_features / z_std, z_mu / z_std) + z_log_var.mean()

        total_loss += loss.item()
        loader_len += 1

    return total_loss / loader_len


def training_loop(
    model,
    batch_size,
    num_input_features,
    num_output_features,
    num_epochs,
    init_lr,
    num_warmup_epochs,
    decay_factor,
    num_feature_swaps,
    enable_gradient_clipping: bool,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params/10**6:.2f}M")

    feature_dim = num_input_features + num_output_features
    weather_indices = torch.arange(feature_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = utils.get_scheduler(optimizer, num_warmup_epochs, decay_factor)

    losses = {
        "train": [],
        "val": [],
    }

    for epoch in range(num_epochs):
        train_loader = utils.streaming_dataloader(
            batch_size,
            shuffle=True,
            split="train",
            lr_finder=False,
        )

        test_loader = utils.streaming_dataloader(
            batch_size,
            shuffle=True,
            split="validation",
            lr_finder=False,
        )

        train_loss = train(
            model,
            num_input_features,
            train_loader,
            weather_indices,
            num_feature_swaps,
            optimizer,
            scheduler,
            criterion,
            DEVICE,
            enable_gradient_clipping,  # Pass the flag to the train function
        )

        if torch.cuda.device_count() > 1:
            model_module = model.model_module
        else:
            model_module = model

        val_loss = validate(
            model_module,  # do the validation on single gpu else error message (no idea why)
            num_input_features,
            test_loader,
            weather_indices,
            num_feature_swaps,
            DEVICE,
        )

        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        logging.info(
            f"Epoch {epoch+1}/{num_epochs}: Losses train: {train_loss:.3f} val: {val_loss:.3f}"
        )
        if epoch % 2 == 1 or epoch == num_epochs - 1:
            torch.save(
                model_module,
                DATA_DIR
                + f"trained_models/weatherformer_{total_params / 10**6:.1f}m_epoch_{epoch}.pth",
            )
            torch.save(
                model_module,
                DATA_DIR
                + f"trained_models/weatherformer_{total_params / 10**6:.1f}m_latest.pth",
            )
            utils.save_losses(losses, total_params, plot=True)
    return model, losses
