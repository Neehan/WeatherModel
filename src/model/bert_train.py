from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from . import utils
from .constants import *


def create_mlm_mask(batch_size, seq_len, mask_percent, device):
    # Calculate the total number of elements in the tensor
    total_elements = batch_size * seq_len

    # Calculate the number of elements to mask
    num_mask = int(mask_percent * total_elements)

    # Generate random indices for masking within the entire tensor
    mask_indices = torch.randperm(total_elements)[:num_mask]

    # Convert flat indices to multi-dimensional indices
    mask_indices = (mask_indices // seq_len, mask_indices % seq_len)

    # Create the initial mask tensor filled with zeros
    mask_tensor = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

    # Set the randomly selected indices to 1
    mask_tensor[mask_indices] = True

    return mask_tensor.bool()


def bert_train(
    model,
    loader,
    mask_pcnt,
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

        batch_size, seq_len, n_features = data.size()
        # true means will be masked
        weather_feature_mask = create_mlm_mask(batch_size, seq_len, mask_pcnt, device)
        # expand the mask
        weather_feature_mask = weather_feature_mask.unsqueeze(2).expand(
            -1, -1, n_features
        )
        target_tokens = data[weather_feature_mask]

        output = model(data, coords, index, weather_feature_mask=weather_feature_mask)[
            weather_feature_mask
        ]
        loss = criterion(target_tokens, output)

        total_loss += loss.item()
        loader_len += 1

        # Backward pass
        loss.backward()
        optimizer.step()
    scheduler.step()

    return np.sqrt(total_loss / loader_len)


def bert_validate(
    model,
    loader,
    mask_pcnt,
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
        batch_size, seq_len, n_features = data.size()
        weather_feature_mask = create_mlm_mask(batch_size, seq_len, mask_pcnt, device)
        # expand the mask
        weather_feature_mask = weather_feature_mask.unsqueeze(2).expand(
            -1, -1, n_features
        )
        target_tokens = data[weather_feature_mask]

        with torch.no_grad():
            output = model(
                data, coords, index, weather_feature_mask=weather_feature_mask
            )[weather_feature_mask]
        loss = F.mse_loss(target_tokens, output)

        total_loss += loss.item()
        loader_len += 1

    return np.sqrt(total_loss / loader_len)


def bert_training_loop(
    model,
    batch_size,
    num_epochs,
    init_lr,
    num_warmup_epochs,
    decay_factor,
    mask_pcnt,
):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params/10**6:.2f}M")

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

        train_loss = bert_train(
            model,
            train_loader,
            mask_pcnt,
            optimizer,
            scheduler,
            criterion,
            DEVICE,
        )

        val_loss = bert_validate(
            model.module,  # do the validation on single gpu else error message (no idea why)
            test_loader,
            mask_pcnt,
            DEVICE,
        )

        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        logging.info(
            f"Epoch {epoch+1}: Losses train: {train_loss:.3f} val: {val_loss:.3f}"
        )
        if epoch % 2 == 1 or epoch == num_epochs - 1:
            torch.save(
                model.module,
                DATA_DIR
                + f"trained_models/bert_{total_params / 10**6:.1f}m_epoch_{epoch}.pth",
            )
            torch.save(
                model.module,
                DATA_DIR
                + f"trained_models/bert_{total_params / 10**6:.1f}m_latest.pth",
            )
            utils.save_losses(losses, total_params)
    return model, losses
