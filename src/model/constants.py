import torch
import logging
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from .tqdm_to_logger import TqdmToLogger

# Automatically finds and loads the .env file
load_dotenv()

plt.style.use("ggplot")

DATA_DIR = "data/"
WEATHER_FILE_PATH = DATA_DIR + "nasa_power/train_dataset_weekly.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the STDOUT environment variable
STDOUT = os.environ.get("STDOUT", "False").lower() in ("true", "1", "t")
# Read the TEST_ENV environment variable
TEST_ENV = os.environ.get("TEST_ENV", "False").lower() in ("true", "1", "t")


if STDOUT:
    # Configure logging to output to stdout
    print("output all logs to stdout")
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
        filename=DATA_DIR + "pretraining.log",
        filemode="w",
    )

TQDM_OUTPUT = TqdmToLogger(logging.getLogger(), level=logging.INFO)
TOTAL_WEATHER_VARS = 31
MAX_GRANULARITY_DAYS = 31
CONTEXT_LENGTH = 365
NUM_DATASET_PARTS = 119
TEST_PART_IDS = [7, 106, 56, 59, 93, 30]
DRY_RUN_TRAIN_PART_IDS = [1, 34, 53, 72, 81]
