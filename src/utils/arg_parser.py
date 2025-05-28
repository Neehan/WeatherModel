import argparse
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", help="batch size", default=64, type=int)
parser.add_argument(
    "--n-input-features", help="number of input features", default=21, type=int
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=75, type=int
)
parser.add_argument("--init-lr", help="initial learning rate", default=1e-6, type=float)
parser.add_argument(
    "--n-warmup-epochs", help="number of warm-up epochs", default=10, type=float
)
parser.add_argument(
    "--decay-factor",
    help="exponential learning rate decay factor after warmup",
    default=0.99,
    type=float,
)
parser.add_argument(
    "--model-size",
    help="model size mini (16k), small (2M), medium (8M), and large (56M)",
    default="mini",
    type=str,
)

parser.add_argument(
    "--model",
    help="model type is weatherformer or bert",
    default="weatherformer",
    type=str,
)

parser.add_argument(
    "--masking-prob",
    help="percent to mask",
    default=0.15,
    type=float,
)


def parse_args():
    args = parser.parse_args()
    args_dict = vars(args)
    logger.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logger.info(f"{arg}: {value}")

    # Model size configuration
    model_size = args.model_size.lower()
    if model_size == "mini":
        model_size_params = {"num_heads": 4, "num_layers": 2, "hidden_dim_factor": 6}
    elif model_size == "small":
        model_size_params = {"num_heads": 10, "num_layers": 4, "hidden_dim_factor": 20}
    elif model_size == "medium":
        model_size_params = {"num_heads": 12, "num_layers": 6, "hidden_dim_factor": 28}
    elif model_size == "large":
        model_size_params = {"num_heads": 16, "num_layers": 8, "hidden_dim_factor": 36}
    args_dict["model_size_params"] = model_size_params
    return args_dict
