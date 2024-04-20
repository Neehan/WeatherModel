import torch
import logging
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from ..model.tqdm_to_logger import TqdmToLogger

# Automatically finds and loads the .env file
load_dotenv()

plt.style.use("ggplot")

DATA_DIR = "data/"
WEATHER_FILE_PATH = DATA_DIR + "nasa_power/train_dataset_weekly.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the STDOUT environment variable
STDOUT = os.environ.get("STDOUT", "False").lower() in ("true", "1", "t")
TEST_ENV = os.environ.get("TEST_ENV", "False").lower() in ("true", "1", "t")

TQDM_OUTPUT = TqdmToLogger(logging.getLogger(), level=logging.INFO)
SEQ_LEN = 52

if STDOUT:
    # Configure logging to output to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
else:
    # Configure logging to write to a file
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=DATA_DIR + "yield_predictor.log",
        filemode="w",
    )

# minimum logging interval in seconds during training
MIN_INTERVAL = 3 * 60
