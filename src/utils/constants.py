import torch
from dotenv import load_dotenv
import os

# Automatically finds and loads the .env file
load_dotenv()

DATA_DIR = "data/"
WEATHER_FILE_PATH = DATA_DIR + "nasa_power/train_dataset_weekly.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the STDOUT environment variable
STDOUT = os.environ.get("STDOUT", "False").lower() in ("true", "1", "t")
# Read the TEST_ENV environment variable
DRY_RUN = os.environ.get("DRY_RUN", "False").lower() in ("true", "1", "t")

# Crop yield statistics (mean and std for normalization)
CROP_YIELD_STATS = {
    "soybean": {
        "mean": [],
        "std": [],
    },
    "corn": {
        "mean": [],
        "std": [],
    },
}

TOTAL_WEATHER_VARS = 31
MAX_GRANULARITY_DAYS = 31
MAX_CONTEXT_LENGTH = 365
NUM_DATASET_PARTS = 119
VALIDATION_CHUNK_IDS = [7, 30, 56, 59, 93, 106, 110, 24]
DRY_RUN_TRAIN_CHUNK_IDS = [1, 34, 53, 72, 81]
