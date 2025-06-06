import torch
import logging
from typing import Dict, Optional, Tuple
from src.pretraining.trainers.weatherformer_trainer import WeatherFormerTrainer
from src.pretraining.models.weatherformer_mixture import WeatherFormerMixture
from src.utils.constants import TOTAL_WEATHER_VARS
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from torch.utils.data import DataLoader
import math
import torch.nn as nn


class WeatherFormerMixtureTrainer(WeatherFormerTrainer):
    """
    WeatherFormerMixture trainer that implements MSE + KL divergence loss.
    """

    def __init__(
        self,
        model: WeatherFormerMixture,
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

    def _compute_mixture_kl_divergence(
        self,
        z: torch.Tensor,  # [batch_size, seq_len, n_features] - sampled latent
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior mean
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features] - posterior variance
        mu_k: torch.Tensor,  # [k, seq_len, n_features] - mixture means
        var_k: torch.Tensor,  # [k, seq_len, n_features] - mixture variances
        feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features] - mask for output features
    ) -> torch.Tensor:
        """
        Compute KL divergence between posterior q(z|x) and mixture prior p(z) for masked features only.

        KL(q(z|x) || p(z)) = log q(z|x) - log p(z)

        where:
        - q(z|x) is a diagonal Gaussian N(mu_x, var_x)
        - p(z) is a mixture of k diagonal Gaussians: (1/k) * Σ_i N(mu_k[i], var_k[i])
        """
        # Get number of mixture components

        # Compute log q(z|x) - posterior log-density for masked features only
        log_q_z_x_all = -0.5 * (torch.log(var_x) + (z - mu_x) ** 2 / var_x)
        log_q_z_x_masked = log_q_z_x_all * feature_mask  # apply mask
        log_q_z_x = torch.sum(log_q_z_x_masked, dim=(1, 2))  # [batch_size]

        # Compute log p(z) - mixture prior log-density for masked features only
        z_expanded = z.unsqueeze(0)  # [1, batch_size, seq_len, n_features]
        mu_k_expanded = mu_k.unsqueeze(1)  # [k, 1, seq_len, n_features]
        var_k_expanded = var_k.unsqueeze(1)  # [k, 1, seq_len, n_features]

        # Compute log-density for each component
        log_component_densities_all = -0.5 * (
            torch.log(var_k_expanded)
            + (z_expanded - mu_k_expanded) ** 2 / var_k_expanded
        )
        # Apply mask to only consider masked features
        log_component_densities_masked = (
            log_component_densities_all * feature_mask.unsqueeze(0)
        )
        log_component_densities = torch.sum(
            log_component_densities_masked, dim=(2, 3)
        )  # [k, batch_size]

        # Compute log p(z) = log(Σ_k exp(log_component_densities))
        log_p_z = torch.logsumexp(log_component_densities, dim=0)  # [batch_size]

        # KL divergence: KL(q(z|x) || p(z)) = log q(z|x) - log p(z)
        kl_divergence = log_q_z_x - log_p_z  # [batch_size]

        return kl_divergence.mean()  # average over batch

    def _masked_mean(
        self, tensor: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]
    ):
        """Mean over `dim`, ignoring False in `mask`."""
        masked = tensor * mask
        return masked.sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1))

    def compute_mixture_loss(
        self,
        weather: torch.Tensor,  # [batch_size, seq_len, n_features]
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features]
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features]
        mu_k: torch.Tensor,  # [k, seq_len, n_features]
        var_k: torch.Tensor,  # [k, seq_len, n_features]
        feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features]
        log_losses: bool = False,
    ):
        """
        Compute loss as Gaussian MSE on masked features + lam * KL divergence on masked features.
        """
        # 1. Gaussian MSE on masked features: -log p(x|mu_x, var_x) for masked features
        # Gaussian NLL: 0.5 * log(var) + 0.5 * (x - mu)^2 / var
        gaussian_nll = 0.5 * torch.log(var_x) + 0.5 * (weather - mu_x) ** 2 / var_x

        # 2. Standard deviation term: std for masked features
        std_term = self._masked_mean(torch.sqrt(var_x), feature_mask, dim=(1, 2)).mean()

        # Apply feature mask and compute mean over masked features
        masked_gaussian_nll = gaussian_nll * feature_mask

        # mean over all dimensions (batch_size, seq_len, n_features)
        reconstruction_loss = self._masked_mean(
            masked_gaussian_nll, feature_mask, dim=(1, 2)
        ).mean()

        # 2. KL divergence term: lam * KL(q(z|x) || p(z)) for masked features only
        # Sample z using reparameterization trick: z = mu_x + sqrt(var_x) * epsilon
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon

        kl_term = self._compute_mixture_kl_divergence(
            z, mu_x, var_x, mu_k, var_k, feature_mask
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
        """Compute WeatherFormerMixture training loss using MSE + KL divergence."""

        # Get model predictions (mu_x, var_x, mu_k, var_k)
        mu_x, var_x, mu_k, var_k = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute mixture loss
        loss_dict = self.compute_mixture_loss(
            weather,
            mu_x,
            var_x,
            mu_k,
            var_k,
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
        """Compute WeatherFormerMixture validation loss using MSE + KL divergence."""

        # Get model predictions (mu_x, var_x, mu_k, var_k)
        mu_x, var_x, mu_k, var_k = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute mixture loss
        loss_dict = self.compute_mixture_loss(
            weather,
            mu_x,
            var_x,
            mu_k,
            var_k,
            feature_mask,
        )

        return loss_dict


def weatherformer_mixture_training_loop(args_dict):
    """
    WeatherFormerMixture training loop using the WeatherFormerMixtureTrainer class.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize WeatherFormerMixture model
    model = WeatherFormerMixture(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        k=args_dict["n_mixture_components"],
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = WeatherFormerMixtureTrainer(
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

    return trainer.train()
