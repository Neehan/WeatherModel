import logging

import torch
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from src.pretraining.models.weatherformer_sinusoid import WeatherFormerSinusoid
from src.pretraining.trainers.weatherformer_trainer import WeatherFormerTrainer
from src.utils.constants import TOTAL_WEATHER_VARS
from src.utils.losses import compute_gaussian_kl_divergence


class WeatherFormerSinusoidTrainer(WeatherFormerTrainer):
    """
    WeatherFormerSinusoid trainer that implements MSE + KL divergence loss.
    KL divergence is between the posterior and sinusoidal prior distributions.
    """

    def __init__(
        self,
        model: WeatherFormerSinusoid,
        masking_prob: float,
        n_masked_features: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(
            model=model,
            masking_prob=masking_prob,
            n_masked_features=n_masked_features,
            beta=beta,  # Use lam as beta in parent class
            **kwargs,
        )

    def compute_kl_loss(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss between posterior and sinusoidal prior distributions."""
        kl_term = compute_gaussian_kl_divergence(
            weather_feature_mask, mu_x, var_x, mu_p, var_p
        )
        return kl_term


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
        k=args_dict["n_mixture_components"],
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
        beta=args_dict["beta"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train(use_optimal_lr=args_dict["use_optimal_lr"])
