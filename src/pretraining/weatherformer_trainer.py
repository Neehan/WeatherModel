import torch
import torch.nn as nn
import logging
import random
from typing import Dict
from src.pretraining.base.base_trainer import BaseTrainer
from src.models.weatherformer import WeatherFormer
from src.utils.constants import TOTAL_WEATHER_VARS, DEVICE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

random.seed(1234)
torch.manual_seed(1234)


class WeatherFormerTrainer(BaseTrainer):
    """
    WeatherFormer trainer that implements feature swapping masking and VAE-style loss.
    """

    def __init__(
        self,
        model: WeatherFormer,
        batch_size: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(model, batch_size, **kwargs)
        self.beta = beta  # Hyperparameter controlling reconstruction vs regularization trade-off

        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "log_variance": [],
                "kl_regularization": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "log_variance": [],
                "kl_regularization": [],
            },
        }

    def get_model_name(self) -> str:
        return "weatherformer"

    def compute_elbo_loss(
        self,
        target_features: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the VAE-style loss from the mathematical formula:
        L_pretrain = mean over masked features of:
        [(z - μ)² / σ²] + (1-β)log σ² + β(μ² + σ²)

        Args:
            target_features: Ground truth values (z in the formula)
            mu: Predicted mean values
            sigma: Predicted standard deviation values
        """
        # Reconstruction term: (z - μ)² / σ²
        reconstruction_term = torch.mean(((target_features - mu) ** 2) / (sigma**2))

        # Log variance term: (1-β) log σ²
        log_variance_term = torch.mean((1 - self.beta) * torch.log(sigma**2))

        # KL regularization term: β(μ² + σ²)
        kl_regularization_term = torch.mean(self.beta * (mu**2 + sigma**2))

        if log_losses:
            self.logger.info(f"Reconstruction Term: {reconstruction_term.item():.6f}")
            self.logger.info(f"Log Variance Term: {log_variance_term.item():.6f}")
            self.logger.info(
                f"KL Regularization Term: {kl_regularization_term.item():.6f}"
            )

        total_loss = reconstruction_term + log_variance_term + kl_regularization_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_term,
            "log_variance": log_variance_term,
            "kl_regularization": kl_regularization_term,
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

        # Get model predictions (mu, sigma)
        mu, sigma = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Extract target features and predictions for masked positions only
        target_features = weather[feature_mask]
        predicted_mu = mu[feature_mask]
        predicted_sigma = sigma[feature_mask]

        # Compute VAE loss
        loss_dict = self.compute_elbo_loss(
            target_features, predicted_mu, predicted_sigma
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

        # Get model predictions (mu, sigma)
        mu, sigma = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Extract target features and predictions for masked positions only
        target_features = weather[feature_mask]
        predicted_mu = mu[feature_mask]
        predicted_sigma = sigma[feature_mask]

        # Compute VAE loss
        loss_dict = self.compute_elbo_loss(
            target_features, predicted_mu, predicted_sigma
        )

        return loss_dict


def weatherformer_training_loop(args_dict):
    """
    WeatherFormer training loop using the WeatherFormerTrainer class.
    Initializes the model internally and handles all training.
    """
    # Initialize WeatherFormer model
    model = WeatherFormer(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=DEVICE,
        **args_dict["model_size_params"],
    ).to(DEVICE)

    logging.info(str(model))

    trainer = WeatherFormerTrainer(
        model=model,
        batch_size=args_dict["batch_size"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        beta=args_dict["beta"],
        masking_function="weatherformer",
        n_masked_features=TOTAL_WEATHER_VARS - args_dict["n_input_features"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
    )

    return trainer.train(args_dict["n_epochs"])
