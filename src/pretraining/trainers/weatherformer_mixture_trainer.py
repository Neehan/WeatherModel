import torch
import logging
from typing import Dict, Optional, Tuple
from src.pretraining.trainers.weatherformer_trainer import WeatherFormerTrainer
from src.pretraining.models.weatherformer_mixture import WeatherFormerMixture
from src.utils.constants import TOTAL_WEATHER_VARS
from src.pretraining.dataloader.pretraining_dataloader import streaming_dataloader
from torch.utils.data import DataLoader
import math


class WeatherFormerMixtureTrainer(WeatherFormerTrainer):
    """
    WeatherFormerMixture trainer that implements Gaussian mixture prior ELBO loss.
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

        # override the losses collected to include mixture prior loss
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "log_variance": [],
                "mixture_prior": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "log_variance": [],
                "mixture_prior": [],
            },
        }
        self.output_json["model_config"]["prior_weight"] = lam

    def _masked_mean(
        self, tensor: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]
    ):
        """Mean over `dim`, ignoring False in `mask`."""
        masked = tensor * mask
        return masked.sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1))

    def gaussian_nll(self, z, mu, var):
        reconstruction = 0.5 * (z - mu) ** 2 / var  # no ½ log 2π
        log_variance = 0.5 * torch.log(var)
        return reconstruction, log_variance

    def compute_elbo_loss(
        self,
        z: torch.Tensor,  # [batch_size, seq_len, n_features]
        mu_x: torch.Tensor,  # [batch_size, seq_len, n_features]
        var_x: torch.Tensor,  # [batch_size, seq_len, n_features]
        mu_k: torch.Tensor,  # [k, seq_len, n_features]
        var_k: torch.Tensor,  # [k, seq_len, n_features]
        feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features]
        log_losses: bool = False,
    ):
        # ---------- encoder term  ----------
        reconstruction, log_variance = self.gaussian_nll(z, mu_x, var_x)  # []
        enc_loss = self._masked_mean(
            reconstruction + log_variance, feature_mask, dim=(1, 2)
        )  # mean over masked seq_len, n_features
        enc_loss = enc_loss.mean()  # then mean over batch_size
        mse_loss = self._masked_mean((z - mu_x) ** 2, feature_mask, dim=(1, 2)).mean()

        # # ---------- mixture‑prior term ----------
        # z = z.unsqueeze(0)  # [1,batch_size,seq_len,n_features]
        # mu_k = mu_k.unsqueeze(1)  # [K,1,seq_len,n_features]
        # var_k = var_k.unsqueeze(1)  # [K,1,seq_len,n_features]

        # nll = self.gaussian_nll(z, mu_k, var_k)  # [K,batch_size,seq_len,n_features]
        # comp_logp = -(nll[0] + nll[1])
        # # mask out input features
        # comp_logp = comp_logp * feature_mask.unsqueeze(0)
        # logp = torch.logsumexp(comp_logp.sum(dim=(2, 3)), dim=0)  # [batch_size,]
        # mix_loss = (-logp).mean() * self.lam
        # var_reg = (
        #     0.3 * self._masked_mean(var_x, feature_mask, dim=(1, 2)).mean()
        # )  # penalize large var
        mix_loss = torch.tensor([0.0])
        total = enc_loss * 0.0 + mse_loss  # + var_reg
        if log_losses:
            self.logger.info(f"Encoder Loss: {enc_loss.item():.6f}")
            self.logger.info(f"Mixture Prior Loss: {mix_loss.item():.6f}")
            # self.logger.info(f"Var Reg Loss: {var_reg.item():.6f}")
            self.logger.info(f"Total Loss: {total.item():.6f}")

        return dict(
            total_loss=total,
            reconstruction=self._masked_mean(
                reconstruction, feature_mask, dim=(1, 2)
            ).mean(),
            log_variance=self._masked_mean(
                log_variance, feature_mask, dim=(1, 2)
            ).mean(),
            mixture_prior=mix_loss,
        )

    def compute_train_loss(
        self,
        weather: torch.Tensor,  # batch_size x seq_len x n_features
        coords: torch.Tensor,  # batch_size x 2
        year: torch.Tensor,  # batch_size x seq_len
        interval: torch.Tensor,  # batch_size
        feature_mask: torch.Tensor,  # batch_size x seq_len x n_features
    ) -> Dict[str, torch.Tensor]:
        """Compute WeatherFormerMixture training loss using Gaussian mixture prior ELBO."""

        # Get model predictions (mu_x, var_x, mu_k, var_k)
        mu_x, var_x, mu_k, var_k = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute mixture ELBO loss with full tensors and mask
        loss_dict = self.compute_elbo_loss(
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
        """Compute WeatherFormerMixture validation loss using Gaussian mixture prior ELBO."""

        # Get model predictions (mu_x, var_x, mu_k, var_k)
        mu_x, var_x, mu_k, var_k = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Compute mixture ELBO loss with full tensors and mask
        loss_dict = self.compute_elbo_loss(
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
