import torch
import logging
from typing import Dict, Optional, Tuple
from src.pretraining.trainers.weatherformer_trainer import WeatherFormerTrainer
from src.pretraining.models.weatherformer_sinusoid import WeatherFormerSinusoid
from src.utils.constants import TOTAL_WEATHER_VARS
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from torch.utils.data import DataLoader
import math
import torch.nn as nn


class WeatherFormerSinusoidTrainer(WeatherFormerTrainer):
    """
    WeatherFormerSinusoid trainer that implements MSE + KL divergence loss.
    KL divergence is between two multivariate normal distributions.
    """

    def __init__(
        self,
        model: WeatherFormerSinusoid,
        masking_prob: float,
        n_masked_features: int,
        lam: float,
        **kwargs,
    ):
        super().__init__(
            model=model,
            masking_prob=masking_prob,
            n_masked_features=n_masked_features,
            **kwargs,
        )
        self.lam = lam
        self.masking_function = "weatherformer"
        self.criterion = nn.MSELoss(reduction="mean")

        # override the losses collected to include KL divergence
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "std_term": [],
                "kl_term": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
                "std_term": [],
            },
        }
        self.output_json["model_config"]["prior_weight"] = lam

    def _compute_gaussian_kl_divergence(
        self,
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_p: torch.Tensor,  # [batch_size, seq_len, n_features] - prior mean
        var_p: torch.Tensor,  # [batch_size, seq_len, n_features] - prior variance
        feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - mask for output features
    ) -> torch.Tensor:
        """
        Compute KL divergence between posterior q(z|x) and sinusoidal prior p(z) for masked features only.

        KL(q(z|x) || p(z)) = 0.5 * [log(var_p/var_x) + var_x/var_p + (mu_x - mu_p)^2/var_p - 1]

        where:
        - q(z|x) is a diagonal Gaussian N(mu_x, var_x)
        - p(z) is a diagonal Gaussian N(mu_p, var_p)
        """
        # Compute KL divergence for each dimension
        kl_per_dim = 0.5 * (
            torch.log(var_p / var_x) + var_x / var_p + (mu_x - mu_p) ** 2 / var_p - 1.0
        )

        # Apply feature mask to only consider masked features
        kl_masked = kl_per_dim * feature_mask

        # Sum over masked features and average over batch
        kl_divergence = self._masked_mean(kl_masked, feature_mask, dim=(1, 2)).mean()

        return kl_divergence

    def compute_sinusoid_loss(
        self,
        weather: torch.Tensor,  # [batch_size, seq_len, n_features]
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features]
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features]
        mu_p: torch.Tensor,  # [batch_size, seq_len, n_features]
        var_p: torch.Tensor,  # [batch_size, seq_len, n_features]
        feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features]
        log_losses: bool = False,
    ):
        """
        Compute loss as Gaussian MSE on masked features + lam * KL divergence on masked features.
        """
        # 1. Gaussian MSE on masked features: -log p(x|mu_x, var_x) for masked features
        # Gaussian NLL: 0.5 * log(var) + 0.5 * (x - mu)^2 / var
        gaussian_nll = 0.5 * torch.log(var_x) + 0.5 * (weather - mu_x) ** 2 / var_x
        # Apply feature mask and compute mean over masked features
        masked_gaussian_nll = gaussian_nll * feature_mask

        # mean over all dimensions (batch_size, seq_len, n_features)
        reconstruction_loss = self._masked_mean(
            masked_gaussian_nll, feature_mask, dim=(1, 2)
        ).mean()

        # 2. Standard deviation term: std for masked features
        std_term = self._masked_mean(torch.sqrt(var_x), feature_mask, dim=(1, 2)).mean()

        # 3. KL divergence term: lam * KL(q(z|x) || p(z)) for masked features only
        kl_term = self._compute_gaussian_kl_divergence(
            mu_x, var_x, mu_p, var_p, feature_mask
        )
        kl_loss = self.lam * kl_term

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        if log_losses:
            self.logger.info(f"Reconstruction Loss: {reconstruction_loss.item():.6f}")
            self.logger.info(f"Std Term: {std_term.item():.6f}")
            self.logger.info(f"KL Loss: {kl_loss.item():.6f}")
            self.logger.info(f"Total Loss: {total_loss.item():.6f}")

        return dict(
            total_loss=total_loss,
            reconstruction=reconstruction_loss,
            kl_term=kl_loss,
            std_term=std_term,
        )

    def compute_train_loss(
        self,
        weather: torch.Tensor,  # batch_size x seq_len x n_features
        coords: torch.Tensor,  # batch_size x 2
        year: torch.Tensor,  # batch_size x seq_len
        interval: torch.Tensor,  # batch_size
        feature_mask: torch.Tensor,  # batch_size x seq_len x n_features
    ) -> Dict[str, torch.Tensor]:
        """Compute WeatherFormerSinusoid training loss using MSE + KL divergence."""

        # Get model predictions (mu_x, var_x, mu_p, var_p)
        mu_x, var_x, mu_p, var_p = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute sinusoid loss
        loss_dict = self.compute_sinusoid_loss(
            weather,
            mu_x,
            var_x,
            mu_p,
            var_p,
            feature_mask,
        )

        return loss_dict

    def compute_validation_loss(
        self,
        weather: torch.Tensor,  # batch_size x seq_len x n_features
        coords: torch.Tensor,  # batch_size x 2
        year: torch.Tensor,  # batch_size x seq_len
        interval: torch.Tensor,  # batch_size
        feature_mask: torch.Tensor,  # batch_size x seq_len x n_features
    ) -> Dict[str, torch.Tensor]:
        """Compute WeatherFormerSinusoid validation loss using MSE + KL divergence."""

        # Get model predictions (mu_x, var_x, mu_p, var_p)
        mu_x, var_x, mu_p, var_p = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute sinusoid loss
        loss_dict = self.compute_sinusoid_loss(
            weather,
            mu_x,
            var_x,
            mu_p,
            var_p,
            feature_mask,
        )

        return loss_dict


def weatherformer_sinusoid_training_loop(args_dict):
    """
    WeatherFormerSinusoid training loop using the WeatherFormerSinusoidTrainer class.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize WeatherFormerSinusoid model
    model = WeatherFormerSinusoid(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = WeatherFormerSinusoidTrainer(
        model=model,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        masking_prob=args_dict["masking_prob"],
        n_masked_features=args_dict["n_masked_features"],
        lam=args_dict["prior_weight"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train(use_optimal_lr=args_dict["use_optimal_lr"])
