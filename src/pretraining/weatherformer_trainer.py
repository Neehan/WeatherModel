import torch
import torch.nn as nn
import logging
import random

from src.pretraining.base.base_trainer import BaseTrainer
from src.utils.arg_parser import parse_args
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
        model,
        batch_size,
        num_input_features,
        num_output_features,
        num_feature_swaps,
        beta,
        **kwargs,
    ):
        super().__init__(model, batch_size, **kwargs)
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.num_feature_swaps = num_feature_swaps
        self.beta = beta  # Hyperparameter controlling reconstruction vs regularization trade-off

        # Initialize feature indices for swapping
        feature_dim = num_input_features + num_output_features
        self.weather_indices = torch.arange(feature_dim)

    def get_model_name(self) -> str:
        return "weatherformer"

    def create_feature_mask(
        self, batch_size: int, seq_len: int, n_features: int
    ) -> torch.Tensor:
        """
        Create feature mask using the swap features strategy.
        This masks entire features across the sequence length dimension.
        """
        # Swap features to create new masking pattern
        self._swap_features(
            self.weather_indices, self.num_input_features, k=self.num_feature_swaps
        )

        # Target indices are the "output" features that we'll predict
        target_indices = self.weather_indices[self.num_input_features :]

        # Create mask tensor - True where features should be masked
        target_mask = torch.zeros(
            TOTAL_WEATHER_VARS, dtype=torch.bool, device=self.device
        )
        target_mask[target_indices] = True

        # Expand mask to match batch and sequence dimensions
        # Shape: (batch_size, seq_len, n_features)
        weather_feature_mask = target_mask.view(1, 1, -1).expand(
            batch_size, seq_len, -1
        )

        return weather_feature_mask

    def _swap_features(self, weather_indices, num_input_features, k=1):
        """
        Swap k indices between input and output features.
        This creates the masking pattern for WeatherFormer training.
        """
        num_output_indices = len(weather_indices) - num_input_features
        k = min(k, num_input_features, num_output_indices)

        # Generate 'k' random indices from the first part of the array (input features)
        input_indices = torch.randperm(num_input_features)[:k]

        # Generate 'k' random indices from the second part of the array (output features)
        output_indices = num_input_features + torch.randperm(num_output_indices)[:k]

        # Perform the swaps
        weather_indices[input_indices], weather_indices[output_indices] = (
            weather_indices[output_indices].clone(),
            weather_indices[input_indices].clone(),
        )
        return weather_indices

    def compute_elbo_loss(
        self,
        target_features: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        log_losses: bool = False,
    ) -> torch.Tensor:
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

        return total_loss

    def compute_train_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WeatherFormer training loss using VAE-style loss function."""
        # Get target indices for the current swap configuration
        target_indices = self.weather_indices[self.num_input_features :]

        # Extract target features (ground truth)
        target_features = weather[:, :, target_indices]

        # Get model predictions (mu, sigma)
        mu, sigma = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Extract predictions for target indices only
        predicted_mu = mu[:, :, target_indices]
        predicted_sigma = sigma[:, :, target_indices]

        # Compute VAE loss
        loss = self.compute_elbo_loss(target_features, predicted_mu, predicted_sigma)

        return loss

    def compute_validation_loss(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WeatherFormer validation loss using VAE-style loss function."""
        # Use a clone of weather_indices to avoid modifying the training state
        validation_indices = self.weather_indices.clone()
        self._swap_features(
            validation_indices, self.num_input_features, k=self.num_feature_swaps
        )

        # Get target indices for validation
        target_indices = validation_indices[self.num_input_features :]

        # Extract target features (ground truth)
        target_features = weather[:, :, target_indices]

        # Get model predictions (mu, sigma)
        mu, sigma = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )

        # Extract predictions for target indices only
        predicted_mu = mu[:, :, target_indices]
        predicted_sigma = sigma[:, :, target_indices]

        # Compute VAE loss
        loss = self.compute_elbo_loss(target_features, predicted_mu, predicted_sigma)

        return loss


def weatherformer_training_loop(
    model,
    batch_size,
    num_input_features,
    num_output_features,
    num_epochs,
    init_lr=1e-4,
    num_warmup_epochs=5,
    decay_factor=0.95,
    num_feature_swaps=1,
    beta=0.1,
):
    """
    WeatherFormer training loop using the WeatherFormerTrainer class.
    """
    trainer = WeatherFormerTrainer(
        model=model,
        batch_size=batch_size,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        init_lr=init_lr,
        num_warmup_epochs=num_warmup_epochs,
        decay_factor=decay_factor,
        num_feature_swaps=num_feature_swaps,
        beta=beta,
    )

    return trainer.train(num_epochs)


if __name__ == "__main__":
    args_dict = parse_args()

    # Initialize WeatherFormer model
    model = WeatherFormer(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=DEVICE,
        **args_dict["model_size_params"],
    ).to(DEVICE)

    logging.info(str(model))

    # Run WeatherFormer training loop with proper parameters
    model, losses = weatherformer_training_loop(
        model=model,
        batch_size=args_dict["batch_size"],
        num_input_features=args_dict["n_input_features"],
        num_output_features=TOTAL_WEATHER_VARS - args_dict["n_input_features"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        num_feature_swaps=args_dict["n_feature_swaps"],
        beta=args_dict.get("beta", 0.1),  # Default beta value if not provided
    )
