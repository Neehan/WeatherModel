import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)


from .dataloader import train_test_split
from .model import FluPredictor
from .train import training_loop
from .constants import *

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", help="batch size", default=64, type=int)
parser.add_argument(
    "--n_past_weeks", help="number of past weeks to look at", default=104, type=int
)

parser.add_argument(
    "--n_future_weeks", help="number of weeks to predict ahead", default=1, type=int
)

parser.add_argument(
    "--n_epochs", help="number of training epoches", default=25, type=int
)
parser.add_argument(
    "--init_lr", help="initial learning rate for Adam", default=0.0005, type=float
)
parser.add_argument(
    "--lr_decay_factor",
    help="learning rate exponential decay factor",
    default=0.98,
    type=float,
)
parser.add_argument(
    "--n_warmup_epochs", help="number of warmup epoches", default=5, type=float
)

parser.add_argument(
    "--no-pretraining",
    dest="no_pretraining",
    action="store_true",
    help="don't use the pretrained model",
)

parser.add_argument(
    "--model_size",
    help="model size small (2M), medium (8M), and large (25M)",
    default="small",
    type=str,
)

parser.set_defaults(no_pretraining=False)


if __name__ == "__main__":

    args = parser.parse_args()

    args_dict = vars(args)
    logging.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logging.info(f"{arg}: {value}")

    # load the datasets
    weather_path = DATA_DIR + "flu_cases/weather_weekly.csv"
    flu_cases_path = DATA_DIR + "flu_cases/flu_cases.json"

    model_size = args.model_size.lower()
    if model_size == "small":
        model_size_params = {"num_heads": 10, "num_layers": 4, "hidden_dim_factor": 20}
        load_model_path = "trained_models/weatherformer_1.9m_latest.pth"
    elif model_size == "medium":
        model_size_params = {"num_heads": 12, "num_layers": 6, "hidden_dim_factor": 28}
        load_model_path = "trained_models/weatherformer_8.2m_latest.pth"
    elif model_size == "large":
        model_size_params = {"num_heads": 16, "num_layers": 8, "hidden_dim_factor": 32}
        load_model_path = "trained_models/weatherformer_25.3m_latest.pth"

    # load the pretrained model
    pretrained_model = (
        None if args.no_pretraining else torch.load(DATA_DIR + load_model_path)
    )

    n_test_years = 5
    total_best_mae = 0
    for test_year in range(2023 - n_test_years, 2023):
        logging.info(f"Testing on year {test_year}")
        model = FluPredictor(pretrained_model, **model_size_params).to(DEVICE)
        train_loader, test_loader = train_test_split(
            weather_path,
            flu_cases_path,
            args.n_past_weeks,
            args.n_future_weeks,
            batch_size=args.batch_size,
            test_year=test_year,
        )
        model, losses, best_mae = training_loop(
            model,
            train_loader,
            test_loader,
            num_epochs=args.n_epochs,
            init_lr=args.init_lr,
            lr_decay_factor=args.lr_decay_factor,
            num_warmup_epochs=args.n_warmup_epochs,
        )
        total_best_mae += best_mae
        del model
    logging.info(f"Average of best MAE: {total_best_mae / n_test_years * 1.73:.3f}")
