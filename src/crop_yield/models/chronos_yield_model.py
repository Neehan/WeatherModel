from typing import Optional, Union
import torch
import torch.nn as nn

import os

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from chronos import ChronosBoltPipeline
from src.pretraining.models.weatherbert import WeatherBERT

from src.base_models.base_model import BaseModel


class ChronosYieldModel(BaseModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        super().__init__(name)
        self.device = device

        # Initialize ChronosBolt components
        self._init_chronos_bolt(device, weather_dim, n_past_years)

        # Attention mechanism to reduce sequence dimension
        self.weather_attention = nn.Sequential(
            nn.Linear(self.total_embedding_dim, 16), nn.GELU(), nn.Linear(16, 1)
        )

        self.yield_mlp = nn.Sequential(
            nn.Linear(
                self.total_embedding_dim + n_past_years + 1, 120
            ),  # total_embedding_dim + past yields
            nn.GELU(),
            nn.Linear(120, 1),
        )

        # Freeze Chronos model by default since it's pretrained
        self.chronos_model_frozen = False
        self.freeze_chronos_model()  # This will set it to True and actually freeze parameters

    def _init_chronos_bolt(self, device, weather_dim, n_past_years):
        """Initialize ChronosBolt pipeline and extract configuration"""
        # Load Chronos-Bolt model for embeddings
        self.chronos_pipeline = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map=device,
            torch_dtype=torch.float32,
        )

        # Ensure all pipeline components are on the correct device
        self.chronos_pipeline.model = self.chronos_pipeline.model.to(device)
        # Register the chronos model as a submodule so it's included in parameters() count
        self.chronos_model = self.chronos_pipeline.model

        # Get embedding dimension from chronos-bolt model
        self.chronos_embedding_dim = self.chronos_pipeline.model.config.d_model

        # Get patch configuration from ChronosBolt
        self.patch_size = self.chronos_pipeline.model.chronos_config.input_patch_size
        self.patch_stride = (
            self.chronos_pipeline.model.chronos_config.input_patch_stride
        )
        self.context_length = self.chronos_pipeline.model.chronos_config.context_length

        # Check if REG token is used
        self.use_reg_token = self.chronos_pipeline.model.chronos_config.use_reg_token
        # Total embedding dim after concatenating all weather features
        self.total_embedding_dim = self.chronos_embedding_dim * weather_dim

    def calculate_num_patches(self, seq_len):
        """
        Calculate number of patches for a given sequence length
        This is computed in the same way as ChronosBolt's Patch class
        """
        # Add padding if needed (same as ChronosBolt)
        padded_len = seq_len
        if seq_len % self.patch_size != 0:
            padded_len = seq_len + (self.patch_size - (seq_len % self.patch_size))
        return (padded_len - self.patch_size) // self.patch_stride + 1

    def get_chronos_embeddings(self, context):
        """
        Extract embeddings from Chronos-Bolt model for weather time series
        """
        embeddings, _, _, _ = self.chronos_pipeline.model.encode(context=context)

        return embeddings

    def yield_model(
        self, weather_embeddings, coord, year, interval, weather_feature_mask, y_past
    ):
        # Apply attention to reduce sequence dimension
        # Compute attention weights
        attention_weights = self.weather_attention(
            weather_embeddings
        )  # batch_size x seq_len x 1
        attention_weights = torch.softmax(
            attention_weights, dim=1
        )  # normalize across sequence

        # Apply attention to get weighted sum
        weather_attended = torch.sum(
            weather_embeddings * attention_weights, dim=1
        )  # batch_size x total_embedding_dim

        mlp_input = torch.cat([weather_attended, y_past], dim=1)
        return self.yield_mlp(mlp_input)

    def load_pretrained(
        self, pretrained_model: Union[WeatherBERT, "ChronosYieldModel"]
    ):
        """
        For Chronos, load_pretrained means we want to unfreeze and fine-tune the model.
        Since Chronos is already pretrained, calling this function will unfreeze it for training.
        """
        self.logger.info(
            f"Chronos load_pretrained called - unfreezing model for fine-tuning"
        )
        # Unfreeze the Chronos model for fine-tuning
        self.unfreeze_chronos_model()

    def forward(self, weather, coord, year, interval, weather_feature_mask, y_past):
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2 (lat, lon) UNNORMALIZED
        year: batch_size x seq_len (UNNORMALIZED, time-varying years)
        interval: batch_size x 1 (UNNORMALIZED in days)
        weather_feature_mask: batch_size x seq_len x n_features
        """

        batch_size, seq_len, weather_dim = weather.shape

        # Reshape to process all weather variables in parallel
        weather_reshaped = weather.permute(0, 2, 1).reshape(
            batch_size * weather_dim, seq_len
        )

        # Get embeddings from ChronosBolt
        embeddings = self.get_chronos_embeddings(weather_reshaped)

        # If REG token is used, exclude it from the embeddings (it's the last token)
        if self.use_reg_token:
            embeddings = embeddings[:, :-1, :]  # Remove last token (REG token)

        num_embeddings = embeddings.shape[1]

        # Reshape embeddings from all weather variables
        embeddings = embeddings.reshape(
            batch_size, weather_dim, num_embeddings, self.chronos_embedding_dim
        )

        # Use attention pooling across the patch dimension instead of assuming seq_len match
        embeddings = embeddings.permute(0, 2, 1, 3).reshape(
            batch_size, num_embeddings, weather_dim * self.chronos_embedding_dim
        )

        output = self.yield_model(
            embeddings,
            coord,
            year,
            interval,
            weather_feature_mask=None,
            y_past=y_past,
        )
        return output

    def freeze_chronos_model(self):
        if not self.chronos_model_frozen:
            self.logger.info("Freezing Chronos model")
            for param in self.chronos_model.parameters():
                param.requires_grad = False
            self.chronos_model_frozen = True

    def unfreeze_chronos_model(self):
        if self.chronos_model_frozen:
            self.logger.info("Unfreezing Chronos model")
            for param in self.chronos_model.parameters():
                param.requires_grad = True
            self.chronos_model_frozen = False
