from typing import Optional, Union
import torch
import torch.nn as nn

import os

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from chronos import ChronosPipeline
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

        # Load Chronos-T5-Tiny model for embeddings
        self.chronos_pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=device,
            torch_dtype=torch.float32,
        )

        # Move all tokenizer tensors to the same device to avoid device mismatch
        tokenizer = self.chronos_pipeline.tokenizer
        for attr_name in dir(tokenizer):
            attr = getattr(tokenizer, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(tokenizer, attr_name, attr.to(device))

        # Store device for dynamic tensor creation
        tokenizer.device = device

        # Monkey patch the _append_eos_token method to use correct device
        def _append_eos_token_fixed(self, token_ids, attention_mask):
            batch_size = token_ids.shape[0]
            eos_tokens = torch.full(
                (batch_size, 1),
                fill_value=self.config.eos_token_id,
                device=token_ids.device,
            )
            token_ids = torch.concat((token_ids, eos_tokens), dim=1)
            eos_mask = torch.full(
                (batch_size, 1), fill_value=True, device=attention_mask.device
            )
            attention_mask = torch.concat((attention_mask, eos_mask), dim=1)
            return token_ids, attention_mask

        tokenizer._append_eos_token = _append_eos_token_fixed.__get__(
            tokenizer, type(tokenizer)
        )

        # Register the chronos model as a submodule so it's included in parameters() count
        self.chronos_model = self.chronos_pipeline.model

        # Get embedding dimension from chronos model
        # chronos-t5-tiny has embedding dim of 256
        self.chronos_embedding_dim = 256
        # Total embedding dim after concatenating all weather features
        self.total_embedding_dim = self.chronos_embedding_dim * weather_dim

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
        self.chronos_model_frozen = (
            False  # Set to False first so freeze_chronos_model() will run
        )
        self.freeze_chronos_model()  # This will set it to True and actually freeze parameters

    def get_chronos_embeddings(self, context):
        """
        Extract embeddings from Chronos model for weather time series
        """

        token_ids, attention_mask, tokenizer_state = (
            self.chronos_pipeline.tokenizer.context_input_transform(context)
        )
        embeddings = self.chronos_pipeline.model.encode(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
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

        # Get embeddings and remove extra token that Chronos adds
        embeddings = self.get_chronos_embeddings(weather_reshaped)
        # Remove extra CLS-like token
        embeddings = embeddings[:, :seq_len, :]

        # Reshape and concatenate embeddings from all weather variables
        embeddings = embeddings.reshape(
            batch_size, weather_dim, seq_len, self.chronos_embedding_dim
        )
        embeddings = embeddings.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, weather_dim * self.chronos_embedding_dim
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
