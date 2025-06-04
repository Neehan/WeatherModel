import torch
import logging
from typing import Dict, Optional, Tuple
from src.base_trainer.base_trainer import BaseTrainer
from src.pretraining.models.weatherformer import WeatherFormer
from src.utils.constants import TOTAL_WEATHER_VARS
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from torch.utils.data import DataLoader
from typing import Tuple


class WeatherFormerTrainer(BaseTrainer):
    """
    WeatherFormer trainer that implements feature swapping masking and VAE-style loss.
    """

    def __init__(
        self,
        model: WeatherFormer,
        masking_prob: float,
        n_masked_features: int,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.masking_prob = masking_prob
        self.n_masked_features = n_masked_features
        self.masking_function = "weatherformer"

        # override the losses collected
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "log_variance": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "log_variance": [],
            },
        }

    def compute_elbo_loss(
        self,
        target_features: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the VAE-style loss from the mathematical formula:
        L_pretrain = mean over masked features of:
        [(z - μ)² / σ²] + log σ²
        Args:
            target_features: Ground truth values (z in the formula)
            mu_x: Predicted mean values
            var_x: Predicted variance values
        """
        # Reconstruction term: (z - μ)² / σ²
        reconstruction_term = torch.mean(((target_features - mu_x) ** 2) / (var_x))

        # Log variance term: log σ²
        log_variance_term = torch.mean(torch.log(var_x))

        if log_losses:
            self.logger.info(f"Reconstruction Term: {reconstruction_term.item():.6f}")
            self.logger.info(f"Log Variance Term: {log_variance_term.item():.6f}")

        total_loss = reconstruction_term + log_variance_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_term,
            "log_variance": log_variance_term,
        }

    def compute_train_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute WeatherFormer training loss using VAE-style loss function."""

        # Get model predictions (mu_x, sigma)
        mu_x, var_x = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Extract target features and predictions for masked positions only
        target_features = weather[feature_mask]
        predicted_mu_x = mu_x[feature_mask]
        predicted_var_x = var_x[feature_mask]

        # Compute VAE loss
        loss_dict = self.compute_elbo_loss(
            target_features, predicted_mu_x, predicted_var_x
        )

        return loss_dict

    def compute_validation_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute WeatherFormer validation loss using VAE-style loss function."""

        # Get model predictions (mu_x, var_x)
        mu_x, var_x = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Extract target features and predictions for masked positions only
        target_features = weather[feature_mask]
        predicted_mu_x = mu_x[feature_mask]
        predicted_var_x = var_x[feature_mask]

        # Compute VAE loss
        loss_dict = self.compute_elbo_loss(
            target_features, predicted_mu_x, predicted_var_x
        )

        return loss_dict

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation."""

        train_loader = streaming_dataloader(
            self.batch_size,
            split="train",
            shuffle=shuffle,
            masking_function=self.masking_function,
            n_masked_features=self.n_masked_features,
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = streaming_dataloader(
            self.batch_size,
            split="validation",
            shuffle=False,
            masking_function=self.masking_function,
            n_masked_features=self.n_masked_features,
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_loader, val_loader


def weatherformer_training_loop(args_dict):
    """
    WeatherFormer training loop using the WeatherFormerTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize WeatherFormer model
    model = WeatherFormer(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = WeatherFormerTrainer(
        model=model,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        masking_prob=args_dict["masking_prob"],
        n_masked_features=args_dict["n_masked_features"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train()
