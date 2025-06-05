import torch
import torch.nn as nn
from src.pretraining.models.weatherformer import WeatherFormer
from src.crop_yield.models.weatherbert_yield_model import WeatherBERTYieldModel
from src.utils.constants import DEVICE, TOTAL_WEATHER_VARS


class WeatherFormerYieldModel(WeatherBERTYieldModel):
    """
    WeatherFormer-based yield prediction model that handles probabilistic weather representations.

    This model extends WeatherBERTYieldModel to work with WeatherFormer's (mu, sigma) output
    and uses the reparameterization trick for sampling weather representations.
    """

    def __init__(
        self,
        name: str,
        mlp_input_dim: int,
        device: torch.device,
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        **model_size_params,
    ):
        # Call parent init but override the weather model
        super().__init__(
            name, mlp_input_dim, device, weather_dim, output_dim, **model_size_params
        )

        # Replace the WeatherBERT with WeatherFormer
        self.weather_model = WeatherFormer(
            weather_dim=weather_dim,
            output_dim=output_dim,
            device=device,
            **model_size_params,
        )

    def forward(self, input_data):
        # Prepare weather input using inherited method
        padded_weather, coord, year, interval, weather_feature_mask = input_data

        # WeatherFormer expects individual arguments, not a tuple
        # and returns (mu, sigma) instead of just weather embeddings
        mu_x, var_x = self.weather_model(
            padded_weather,
            coord,
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
        )

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_x)
        weather_repr = mu_x  # + torch.sqrt(var_x) * epsilon

        # Flatten the weather representation for MLP
        weather_repr = weather_repr.reshape(weather_repr.size(0), -1)

        # Pass through MLP to get yield prediction
        yield_pred = self.mlp(weather_repr)
        return yield_pred, mu_x, var_x
