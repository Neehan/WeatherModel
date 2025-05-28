import torch
from src.crop_yield.base.base_yield_model import BaseYieldPredictor


class WeatherFormerYieldModel(BaseYieldPredictor):
    """
    WeatherFormer-based yield prediction model that handles probabilistic weather representations.

    This model extends BaseYieldPredictor to work with WeatherFormer's (mu, sigma) output
    and uses the reparameterization trick for sampling weather representations.
    """

    def __init__(self, name: str, weather_model, mlp_input_dim: int):
        super().__init__(name, weather_model, mlp_input_dim)

    def forward(self, input_data):
        # Prepare weather input using base class method
        weather, coord, year, interval, weather_feature_mask = (
            self.prepare_weather_input(input_data)
        )

        # WeatherFormer expects individual arguments, not a tuple
        # and returns (mu, sigma) instead of just weather embeddings
        mu, sigma = self.weather_model(
            weather=weather,
            coords=coord,
            year=year,
            interval=interval,
            weather_feature_mask=weather_feature_mask,
        )

        # Apply reparameterization trick: z = mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu)
        weather_repr = mu + sigma * epsilon

        # Flatten the weather representation for MLP
        weather_repr = weather_repr.view(weather_repr.size(0), -1)

        # Pass through MLP to get yield prediction
        output = self.mlp(weather_repr)
        return output
