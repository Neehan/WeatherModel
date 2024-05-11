import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)
import numpy as np

from .dataloader import train_test_split
from .model import FluPredictor
from .linear_model import LinearFluPredictor
from .only_transformer import OnlyTransformerFluPredictor
from .train import training_loop
from .constants import *

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", help="batch size", default=32, type=int)
# parser.add_argument(
#     "--n_past_weeks", help="number of past weeks to look at", default=52 * 3, type=int
# )
parser.add_argument(
    "--year_cutoff", help="cut off data until this year", default=2020, type=int
)


parser.add_argument(
    "--n_predict_weeks", help="number of weeks to predict ahead", default=5, type=int
)
parser.add_argument(
    "--n_eval_weeks", help="number of weeks to evaluate ahead", default=1, type=int
)  # basically you could predict 5 weeks during training, but only evaluate on 1 week advance

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
parser.set_defaults(no_pretraining=False)


parser.add_argument(
    "--model_size",
    help="model size small (2M), medium (8M), and large (25M)",
    default="small",
    type=str,
)

parser.add_argument(
    "--model_type",
    help="weatherformer, linear, transformer",
    default="weatherformer",
    type=str,
)


if __name__ == "__main__":

    args = parser.parse_args()

    args_dict = vars(args)
    logging.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logging.info(f"{arg}: {value}")

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

    best_mae_mean, best_mae_std = 999.0, 999.0

    for n_past_weeks in range(105, 136, 5):
        n_test_years = 4
        total_best_mae = 0
        logging.info(f"\n\nnumber of past weeks: {n_past_weeks}")
        maes = []

        # assert args.n_past_weeks > 52, "need at least one year of past data"
        for test_year in range(args.year_cutoff - n_test_years, args.year_cutoff):
            logging.info(f"Testing on year {test_year}")
            # load the pretrained model
            model_type = args.model_type.lower()
            if model_type == "weatherformer":
                pretrained_model = (
                    None
                    if args.no_pretraining
                    else torch.load(DATA_DIR + load_model_path)
                )

                model = FluPredictor(
                    pretrained_model, model_size_params, args.n_predict_weeks
                ).to(DEVICE)
            elif model_type == "linear":
                model = LinearFluPredictor(
                    args.n_past_weeks * 35, args.n_predict_weeks
                ).to(DEVICE)
            elif model_type == "transformer":
                model = OnlyTransformerFluPredictor(
                    input_dim=35, n_predict_weeks=args.n_predict_weeks
                ).to(DEVICE)
            # load the datasets
            weather_path = DATA_DIR + "flu_cases/weather_weekly.csv"
            flu_cases_path = DATA_DIR + "flu_cases/flu_cases.json"

            train_loader, test_loader = train_test_split(
                weather_path,
                flu_cases_path,
                n_past_weeks,
                args.n_predict_weeks,
                batch_size=args.batch_size,
                test_year=test_year,
            )
            losses, best_mae = training_loop(
                model,
                train_loader,
                test_loader,
                num_epochs=args.n_epochs,
                init_lr=args.init_lr,
                lr_decay_factor=args.lr_decay_factor,
                num_warmup_epochs=args.n_warmup_epochs,
                n_eval_weeks=args.n_eval_weeks,
            )
            maes.append(best_mae)
        current_maes_mean = np.mean(maes)
        current_maes_std = np.std(maes)
        target_std = 2.43
        logging.info(
            f"n_past_weeks {n_past_weeks}, best MAE mean: {current_maes_mean * target_std:.3f}, std: {current_maes_std * target_std:.3f}"
        )
        best_mae_mean = min(best_mae_mean, current_maes_mean)
        best_mae_std = min(best_mae_std, current_maes_std)

    logging.info(
        f"best overall MAE mean: {best_mae_mean * target_std:.3f}, std: {best_mae_std * target_std:.3f}"
    )
