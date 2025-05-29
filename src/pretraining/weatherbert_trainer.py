from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.pretraining.base.base_trainer import BaseTrainer
from src.utils.arg_parser import parse_args
from src.models.weatherbert import WeatherBERT
from src.utils.constants import TOTAL_WEATHER_VARS, DEVICE

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

    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.mse_loss = nn.MSELoss()

    def get_model_name(self) -> str:
        return "weatherbert"

    def create_feature_mask(
        self, batch_size: int, seq_len: int, n_features: int
    ) -> torch.Tensor:
        """
        Create a BERT-like MLM mask for the weather features.
        Randomly mask individual tokens across batch×seq_len×features space.
        """
        # Calculate the total number of elements across all dimensions
        total_elements = batch_size * seq_len * n_features

        # Calculate the number of elements to mask
        num_mask = int(self.masking_prob * total_elements)

        # Generate random indices for masking within the entire flattened tensor
        mask_indices = torch.randperm(total_elements)[:num_mask]

        # Convert flat indices to multi-dimensional indices
        batch_indices = mask_indices // (seq_len * n_features)
        remaining = mask_indices % (seq_len * n_features)
        seq_indices = remaining // n_features
        feature_indices = remaining % n_features

        # Create the mask tensor filled with False
        weather_feature_mask = torch.zeros(
            batch_size, seq_len, n_features, dtype=torch.bool, device=self.device
        )

        # Set the randomly selected indices to True
        weather_feature_mask[batch_indices, seq_indices, feature_indices] = True

        return weather_feature_mask

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
        loss = self.mse_loss(target_tokens, output)

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
        loss = F.mse_loss(target_tokens, output)

        return {"total_loss": loss}


def bert_training_loop(
    model,
    batch_size,
    num_epochs,
    init_lr=1e-4,
    num_warmup_epochs=5,
    decay_factor=0.95,
    masking_prob=0.15,
):
    """
    Simplified BERT training loop using the BertTrainer class.
    """
    trainer = WeatherBertTrainer(
        model=model,
        batch_size=batch_size,
        init_lr=init_lr,
        num_warmup_epochs=num_warmup_epochs,
        decay_factor=decay_factor,
        masking_prob=masking_prob,
        masking_function="weatherbert",
    )

    return trainer.train(num_epochs)


if __name__ == "__main__":
    args_dict = parse_args()

    # Initialize WeatherBERT model
    model = WeatherBERT(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=DEVICE,
        **args_dict["model_size_params"],
    ).to(DEVICE)

    logging.info(str(model))
    # Run BERT training loop with proper parameters
    model, losses = bert_training_loop(
        model=model,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        masking_prob=args_dict["masking_prob"],
    )
