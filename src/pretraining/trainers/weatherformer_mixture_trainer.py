import logging
from typing import Dict

import torch
import torch.nn as nn

from src.pretraining.models.weatherformer import WeatherFormer
from src.pretraining.models.weatherformer_mixture import WeatherFormerMixture
from src.pretraining.trainers.weatherformer_trainer import WeatherFormerTrainer
from src.utils.constants import TOTAL_WEATHER_VARS
from src.utils.losses import compute_mixture_kl_divergence


class WeatherFormerMixtureTrainer(WeatherFormerTrainer):
    """
    WeatherFormerMixture trainer that uses mixture prior for KL divergence.
    """

    def __init__(
        self,
        model: WeatherFormer,
        masking_prob: float,
        n_masked_features: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(
            model=model,
            masking_prob=masking_prob,
            n_masked_features=n_masked_features,
            beta=beta,
            **kwargs,
        )

    def compute_kl_loss(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_k: torch.Tensor,
        var_k: torch.Tensor,
        log_w_k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss using mixture prior with learnable weights."""
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon
        kl_term = compute_mixture_kl_divergence(
            z=z,
            feature_mask=weather_feature_mask,
            mu_x=mu_x,
            var_x=var_x,
            mu_k=mu_k,
            var_k=var_k,
            log_w_k=log_w_k,
        )
        return kl_term


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
        beta=args_dict["beta"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train(use_optimal_lr=args_dict["use_optimal_lr"])
