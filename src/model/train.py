from tqdm import tqdm
import numpy as np
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
        data,
        coords,
        index,
    ) in loader:
        data, coords, index = data.to(device), coords.to(device), index.to(device)
        optimizer.zero_grad()

        # swap some input and output feature of weather indices in place
        swap_features(weather_indices, num_input_features, k=num_feature_swaps)
        target_indices = weather_indices[num_input_features:]
        target_features = data[:, :, target_indices]
        target_mask = torch.zeros(TOTAL_WEATHER_VARS, dtype=torch.bool, device=DEVICE)
        target_mask[target_indices] = True
        target_mask = target_mask.view(1, -1).expand(data.shape[0], -1)

        output = model(data, coords, index, weather_feature_mask=target_mask)[
            :, :, target_indices
        ]

        loss = criterion(target_features, output)

        total_loss += loss.item()
        loader_len += 1

        # Backward pass
        loss.backward()

        if enable_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
    scheduler.step()

    return np.sqrt(total_loss / loader_len)


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
        data,
        coords,
        index,
    ) in loader:
        data, coords, index = data.to(device), coords.to(device), index.to(device)
        # swap one input and output feature of weather indices in place
        swap_features(weather_indices, num_input_features, k=num_feature_swaps)
        target_indices = weather_indices[num_input_features:]
        target_features = data[:, :, target_indices]
        target_mask = torch.zeros(TOTAL_WEATHER_VARS, dtype=torch.bool, device=DEVICE)
        target_mask[target_indices] = True
        target_mask = target_mask.view(1, -1).expand(data.shape[0], -1)

        with torch.no_grad():
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
    num_feature_swaps,
    enable_gradient_clipping: bool,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params/10**6:.2f}M")

    feature_dim = num_input_features + num_output_features
    weather_indices = torch.arange(feature_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    data_loader_dir = DATA_DIR + "nasa_power/pytorch/"

    if TEST_ENV:
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

    # Find the optimal learning rate
    train_loader = utils.streaming_dataloader(
        train_loader_paths, batch_size, shuffle=True, split="train"
    )
    if TEST_ENV:
        optimal_lr = utils.find_optimal_lr(
            model, criterion, optimizer, train_loader, DEVICE
        )
    logging.info(f"optimal learning rate: {optimal_lr:.6f}")

    # Update the optimizer with the optimal learning rate
    optimizer = optim.Adam(model.parameters(), lr=optimal_lr)

    scheduler = utils.get_scheduler(optimizer, num_warmup_epochs, decay_factor)

    losses = {
        "train": [],
        "val": [],
    }

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
            f"Epoch {epoch+1}: Losses train: {train_loss:.3f} val: {val_loss:.3f}"
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
