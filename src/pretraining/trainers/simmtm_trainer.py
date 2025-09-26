import torch
import logging
from src.pretraining.trainers.weatherbert_trainer import WeatherBertTrainer
from src.pretraining.models.simmtm import SimMTM
from src.utils.constants import TOTAL_WEATHER_VARS


class SimMTMTrainer(WeatherBertTrainer):
    """
    SimMTM trainer that inherits from WeatherBertTrainer.
    Uses the same MSE loss and training logic, but with SimMTM masking function.
    Implements contiguous segment masking with geometric mean length of 5.
    """

    def __init__(
        self,
        model: SimMTM,
        masking_prob: float,
        n_masked_features: int,
        **kwargs,
    ):
        # Call parent constructor with the model
        super().__init__(model, masking_prob, n_masked_features, **kwargs)

        # Override masking function to use SimMTM masking
        self.masking_function = "simmtm"


def simmtm_training_loop(args_dict):
    """
    SimMTM training loop using the SimMTMTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize SimMTM model
    model = SimMTM(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = SimMTMTrainer(
        model=model,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        masking_prob=args_dict["masking_prob"],
        n_masked_features=args_dict["n_masked_features"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    return trainer.train(use_optimal_lr=args_dict["use_optimal_lr"])
