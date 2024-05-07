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
    "--n_past_weeks", help="number of past years to look at", default=52, type=int
)
parser.add_argument(
    "--n_epochs", help="number of training epoches", default=20, type=int
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
    "--n_warmup_epochs", help="number of warmup epoches", default=4, type=float
)

parser.add_argument(
    "--load-model",
    help="load pretrained model",
    default="trained_models/weatherformer_4.8m_latest.pth",
    type=str,
)

parser.add_argument(
    "--no-pretraining",
    dest="no_pretraining",
    action="store_true",
    help="don't use the pretrained model",
)
parser.set_defaults(no_pretraining=False)


if __name__ == "__main__":

    args = parser.parse_args()

    args_dict = vars(args)
    logging.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logging.info(f"{arg}: {value}")

    # load the datasets
    weather_path = DATA_DIR + "weather_weekly.csv"
    flu_cases_path = DATA_DIR + "flu_cases.json"
    train_loader, test_loader = train_test_split(
        weather_path, flu_cases_path, args.n_past_weeks, batch_size=args.batch_size
    )

    # load the pretrained model
    pretrained_model = (
        None if args.no_pretraining else torch.load(DATA_DIR + args.load_model)
    )

    model = FluPredictor(pretrained_model).to(DEVICE)
    model, losses = training_loop(
        model,
        train_loader,
        test_loader,
        num_epochs=args.n_epochs,
        init_lr=args.init_lr,
        lr_decay_factor=args.lr_decay_factor,
        num_warmup_epochs=args.n_warmup_epochs,
    )


if __name__ == "__main__":

    _ = read_data(weather_columns, weather_path, flu_cases_path)
