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
    "wheat": {
        "mean": [],
        "std": [],
    },
    "sunflower": {
        "mean": [],
        "std": [],
    },
    "cotton": {
        "mean": [],
        "std": [],
    },
    "sugarcane": {
        "mean": [],
        "std": [],
    },
    "beans": {
        "mean": [],
        "std": [],
    },
    "beans_rainfed": {
        "mean": [],
        "std": [],
    },
    "beans_irrigated": {
        "mean": [],
        "std": [],
    },
    "corn_rainfed": {
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

# Test years for cross-validation
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]

EXTREME_YEARS = {
    "usa": {
        "corn": [2002, 2004, 2009, 2012, 2014],
        "soybean": [2003, 2004, 2009, 2012, 2016],
    },
    "argentina": {
        "corn": [2004, 2005, 2007, 2009, 2015],
        "soybean": [2003, 2006, 2007, 2009, 2015],
        "wheat": [2002, 2003, 2005, 2009, 2011],
        "sunflower": [2002, 2007, 2008, 2009, 2011],
    },
    "brazil": {
        "corn": [2001, 2003, 2007, 2010, 2015],
        "soybean": [2001, 2003, 2005, 2011, 2017],
        "sugarcane": [2002, 2003, 2008, 2012, 2017],
        "wheat": [2001, 2003, 2010, 2015, 2016],
        "cotton": [2004, 2008, 2013, 2017, 2018],
    },
    "mexico": {
        "beans": [2016, 2017, 2018, 2021, 2023],
        "beans_rainfed": [2013, 2014, 2017, 2018, 2021],
        "corn": [2014, 2017, 2019, 2022, 2023],
        "corn_rainfed": [2014, 2017, 2021, 2023, 2024],
        "sugarcane": [2013, 2014, 2018, 2020, 2021],
        "wheat": [2013, 2021, 2022, 2023, 2024],
    },
}
