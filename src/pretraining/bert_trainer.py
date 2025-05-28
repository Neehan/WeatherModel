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


class BertTrainer(BaseTrainer):
    """
    BERT-style trainer that implements masked language modeling for weather data.
    """

    def __init__(self, model, batch_size, mask_percent=0.15, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.mask_percent = mask_percent
        self.criterion = nn.MSELoss()

    def get_model_name(self) -> str:
        return "weatherbert"

    def create_feature_mask(
        self, batch_size: int, seq_len: int, n_features: int
    ) -> torch.Tensor:
        """
        Create a BERT-like MLM mask for the weather features.
        Randomly mask a percentage of the weather features.
        """
        # Calculate the total number of elements in the feature dimension
        total_elements = batch_size * n_features

        # Calculate the number of elements to mask
        num_mask = int(self.mask_percent * total_elements)

        # Generate random indices for masking within the entire tensor
        mask_indices = torch.randperm(total_elements)[:num_mask]

        # Convert flat indices to multi-dimensional indices
        mask_indices = (mask_indices // n_features, mask_indices % n_features)

        # Create the initial mask tensor filled with zeros
        mask_tensor = torch.zeros(
            batch_size, n_features, dtype=torch.float32, device=self.device
        )

        # Set the randomly selected indices to 1
        mask_tensor[mask_indices] = True

        # Expand the mask to match the sequence dimension
        weather_feature_mask = mask_tensor.unsqueeze(1).expand(-1, seq_len, -1)

        return weather_feature_mask.bool()

    def compute_train_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        index: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BERT training loss using MSE between predicted and actual masked tokens."""
        target_tokens = data[feature_mask]

        output = self.model(data, coords, index, weather_feature_mask=feature_mask)[
            feature_mask
        ]
        loss = self.criterion(target_tokens, output)

        return loss

    def compute_validation_loss(
        self,
        data: torch.Tensor,
        coords: torch.Tensor,
        index: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BERT validation loss using MSE between predicted and actual masked tokens."""
        target_tokens = data[feature_mask]

        output = self.model(data, coords, index, weather_feature_mask=feature_mask)[
            feature_mask
        ]
        loss = F.mse_loss(target_tokens, output)

        return loss


def bert_training_loop(
    model,
    batch_size,
    num_epochs,
    init_lr=1e-4,
    num_warmup_epochs=5,
    decay_factor=0.95,
    mask_percent=0.15,
):
    """
    Simplified BERT training loop using the BertTrainer class.
    """
    trainer = BertTrainer(
        model=model,
        batch_size=batch_size,
        init_lr=init_lr,
        num_warmup_epochs=num_warmup_epochs,
        decay_factor=decay_factor,
        mask_percent=mask_percent,
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
        mask_percent=args_dict["mask_pcnt"],
    )
