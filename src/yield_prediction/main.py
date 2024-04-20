import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

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
    "--n_epochs", help="number of training epoches", default=20, type=int
)
parser.add_argument(
    "--init_lr", help="initial learning rate for Adam", default=0.0009, type=float
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
parser.set_defaults(no_pretraining=False)


if __name__ == "__main__":

    args = parser.parse_args()

    # load the datasets
    soybean_df = read_soybean_dataset(DATA_DIR)
    train_loader, test_loader = get_train_test_loaders(
        soybean_df, n_past_years=args.n_past_years, batch_size=args.batch_size
    )

    # load the pretrained model
    pretrained_model = (
        None
        if args.no_pretraining
        else torch.load(DATA_DIR + "trained_models/weatherformer_0.2m_epoch_4.pth")
    )

    model = YieldPredictor(pretrained_model).to(DEVICE)
    model, losses = training_loop(
        model,
        train_loader,
        test_loader,
        num_epochs=args.n_epochs,
        init_lr=args.init_lr,
        lr_decay_factor=args.lr_decay_factor,
        num_warmup_epochs=args.n_warmup_epochs,
    )
