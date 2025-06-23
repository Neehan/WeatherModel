import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.base_trainer.base_trainer import BaseTrainer
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from src.pretraining.models.weatherformer import WeatherFormer
from src.utils.constants import DRY_RUN, TOTAL_WEATHER_VARS
from src.utils.losses import compute_gaussian_kl_divergence, gaussian_log_likelihood


class WeatherFormerTrainer(BaseTrainer):
    """
    WeatherFormer trainer that implements feature  swapping masking and VAE-style loss
    between a diagonal Gaussian posterior and a standard normal prior.
    """

    def __init__(
        self,
        model: WeatherFormer,
        masking_prob: float,
        n_masked_features: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.masking_prob = masking_prob
        self.n_masked_features = n_masked_features
        self.masking_function = "weatherformer"
        self.beta = beta
        self.output_json["model_config"]["beta"] = beta
        # override the losses collected
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
        }

    def compute_kl_loss(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        """Compute KL divergence loss between two multivariate normal distributions.
        Some derived classes may need to override this method."""

        # Standard normal prior for VAE: mu_p = 0, var_p = 1
        mu_p = torch.zeros_like(mu_x)
        var_p = torch.ones_like(var_x)

        kl_term = compute_gaussian_kl_divergence(
            weather_feature_mask, mu_x, var_x, mu_p, var_p
        )
        return kl_term

    def compute_elbo_loss(
        self,
        weather: torch.Tensor,
        feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        *args,
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the correct Gaussian negative log-likelihood loss.

        The negative log-likelihood for a Gaussian is:
        NLL = 0.5 * log(2π) + 0.5 * log(σ²) + (z-μ)²/(2σ²)

        We drop the constant 0.5 * log(2π) term since it doesn't affect optimization.

        Args:
            target_features: Ground truth values (z in the formula)
            mu_x: Predicted mean values
            var_x: Predicted variance values (σ²) - already clamped in model
        """
        # Reconstruction term: (z - μ)² / (2σ²) + 1/2log(σ²)
        n_masked_features = feature_mask.sum(dim=(1, 2)).float().mean()
        reconstruction_term = (
            -gaussian_log_likelihood(weather, mu_x, var_x, feature_mask)
            / n_masked_features
        ).mean()
        kl_term = (
            self.beta
            * self.compute_kl_loss(weather, feature_mask, mu_x, var_x, *args).mean()
        ) / n_masked_features

        if log_losses or DRY_RUN:
            self.logger.info(f"Reconstruction Term: {reconstruction_term.item():.6f}")
            self.logger.info(f"KL Term: {kl_term.item():.6f}")
        # Total loss: reconstruction + log_variance
        total_loss = reconstruction_term + kl_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_term,
            "kl_term": kl_term,
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

        # Get model predictions - first two returns are always mu_x and var_x
        model_outputs = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute VAE loss - pass all outputs directly
        loss_dict = self.compute_elbo_loss(weather, feature_mask, *model_outputs)

        return loss_dict

    def compute_validation_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute WeatherFormer validation loss using VAE-style loss function. Some derived classes
        need to override this method."""

        # Get model predictions - first two returns are always mu_x and var_x
        model_outputs = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute loss - pass all outputs directly
        loss_dict = self.compute_elbo_loss(weather, feature_mask, *model_outputs)
        return loss_dict

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation."""

        current_n_masked = self._get_n_masked_features(
            self.current_epoch, self.n_masked_features
        )

        train_loader = streaming_dataloader(
            self.batch_size,
            split="train",
            shuffle=shuffle,
            masking_function=self.masking_function,
            n_masked_features=current_n_masked,
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = streaming_dataloader(
            self.batch_size,
            split="validation",
            shuffle=False,
            masking_function=self.masking_function,
            n_masked_features=current_n_masked,
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
        beta=args_dict["beta"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train(use_optimal_lr=args_dict["use_optimal_lr"])
