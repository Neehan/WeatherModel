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

# USA grids are 0-34, excluding them from training/validation (using as test set)
# Only use non-USA grids 35-118 for training and validation
USA_CHUNK_IDS = list(range(35))  # grids 0-34 (reserved for testing)
NON_USA_CHUNK_IDS = list(range(35, NUM_DATASET_PARTS))  # grids 35-118

# Updated validation chunks - all from non-USA region (35-118)
# Replaced USA chunks (7, 30, 24) with non-USA chunks (42, 68, 89)
VALIDATION_CHUNK_IDS = [45, 56, 59, 76, 93, 106, 110, 118]
DRY_RUN_TRAIN_CHUNK_IDS = [36, 45, 53, 72, 81]  # Updated to non-USA chunks
