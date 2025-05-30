from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

from src.base_trainer.base_trainer import BaseTrainer
from src.base_models.weatherbert import WeatherBERT
from src.utils.constants import TOTAL_WEATHER_VARS
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from torch.utils.data import DataLoader
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import random

random.seed(1234)
torch.manual_seed(1234)


class WeatherBertTrainer(BaseTrainer):
    """
    BERT-style trainer that implements masked language modeling for weather data.
    """

    def __init__(
        self,
        model: WeatherBERT,
        masking_prob: float,
        n_masked_features: int,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.criterion = nn.MSELoss(reduction="mean")
        self.masking_function = "weatherbert"
        self.masking_prob = masking_prob
        self.n_masked_features = n_masked_features

        self.output_json["model_config"]["masking_function"] = self.masking_function
        self.output_json["model_config"]["masking_prob"] = masking_prob
        self.output_json["model_config"]["n_masked_features"] = n_masked_features

    def compute_train_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute BERT training loss using MSE between predicted and actual masked tokens."""
        target_tokens = data[feature_mask]

        output = self.model(
            data, coords, year, interval, weather_feature_mask=feature_mask
        )[feature_mask]
        loss = self.criterion(target_tokens, output)

        return {"total_loss": loss}

    def compute_validation_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute BERT validation loss using MSE between predicted and actual masked tokens."""
        target_tokens = data[feature_mask]

        output = self.model(
            data, coords, year, interval, weather_feature_mask=feature_mask
        )[feature_mask]
        # Use consistent loss function (self.mse_loss instead of F.mse_loss)
        loss = self.criterion(target_tokens, output)

        return {"total_loss": loss}

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation."""

        train_loader = streaming_dataloader(
            self.batch_size,
            split="train",
            shuffle=shuffle,
            masking_function=self.masking_function,
            masking_prob=self.masking_prob,
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = streaming_dataloader(
            self.batch_size,
            split="validation",
            shuffle=False,
            masking_function=self.masking_function,
            masking_prob=self.masking_prob,
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_loader, val_loader


def weatherbert_training_loop(args_dict):
    """
    BERT training loop using the WeatherBertTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize WeatherBERT model
    model = WeatherBERT(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = WeatherBertTrainer(
        model=model,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        masking_prob=args_dict["masking_prob"],
        n_masked_features=args_dict["n_masked_features"],
        resume_from_checkpoint=args_dict["resume_from_checkpoint"],
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train()
