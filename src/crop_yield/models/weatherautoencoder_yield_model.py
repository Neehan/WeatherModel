import torch
from src.pretraining.models.weatherautoencoder import WeatherAutoencoder
from src.crop_yield.models.weatherbert_yield_model import WeatherBERTYieldModel
from src.utils.constants import TOTAL_WEATHER_VARS


class WeatherAutoencoderYieldModel(WeatherBERTYieldModel):
    def __init__(
        self,
        name: str,
        device: torch.device,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        # Call parent constructor but override the weather model
        super().__init__(name, device, weather_dim, n_past_years, **model_size_params)

        # Replace WeatherBERT with WeatherAutoencoder
        self.weather_model = WeatherAutoencoder(
            weather_dim=weather_dim,
            output_dim=weather_dim,
            device=device,
            **model_size_params,
        )

    def load_pretrained(self, pretrained_model: WeatherAutoencoder):
        """
        Load pretrained WeatherAutoencoder model
        """
        self.weather_model.load_pretrained(pretrained_model)
