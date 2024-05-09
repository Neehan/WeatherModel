import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np

torch.use_deterministic_algorithms(True)


from .dataloader import read_soybean_dataset, get_train_test_loaders
from .model import YieldPredictor
from .train import training_loop
from .constants import *

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", help="batch size", default=64, type=int)
parser.add_argument(
    "--n_past_years", help="number of past years to look at", default=6, type=int
)
parser.add_argument(
    "--n_epochs", help="number of training epoches", default=30, type=int
)
parser.add_argument(
    "--init_lr", help="initial learning rate for Adam", default=0.0006, type=float
)
parser.add_argument(
    "--lr_decay_factor",
    help="learning rate exponential decay factor",
    default=0.95,
    type=float,
)
parser.add_argument(
    "--n_warmup_epochs", help="number of warmup epoches", default=4, type=float
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
    soybean_df = read_soybean_dataset(DATA_DIR)
    soybean_states = set(soybean_df["State"].values)

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
    model = YieldPredictor(pretrained_model, **model_size_params).to(DEVICE)

    total_best_val_loss = 0

    for n in range(5):
        test_states = np.random.choice(
            np.array(list(soybean_states)), size=2, replace=False
        )
        logging.info(f"Testing on: {test_states}")
        train_loader, test_loader = get_train_test_loaders(
            soybean_df,
            test_states,
            n_past_years=args.n_past_years,
            batch_size=args.batch_size,
        )

        model, losses, best_val_loss = training_loop(
            model,
            train_loader,
            test_loader,
            num_epochs=args.n_epochs,
            init_lr=args.init_lr,
            lr_decay_factor=args.lr_decay_factor,
            num_warmup_epochs=args.n_warmup_epochs,
        )
        total_best_val_loss += best_val_loss

    print(f"Average of best val. loss: {total_best_val_loss/5*11.03:3f} Bu/Acre.")
