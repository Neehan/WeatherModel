import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from grids import GRID
from constants import *

DATA_DIR = "data/nasa_power"

# USA regions only (file IDs 0-33 based on the current dataloader pattern)
USA_REGIONS = [f"usa_{i}" for i in range(len(GRID["USA"]))]

# Central America regions only (file IDs 34-43 based on the current dataloader pattern)
CENTRAL_AMERICA_REGIONS = [
    f"centralamerica_{i}" for i in range(len(GRID["CENTRALAMERICA"]))
]
# SOUTH_AMERICA_REGIONS = [f"southamerica_{i}" for i in range(len(GRID["SOUTHAMERICA"]))]


def process_regions(region_list, region_name, subdirectory, weekly_data):
    """
    Process weather data for a list of regions and accumulate weekly statistics.

    Args:
        region_list: List of region IDs to process
        region_name: Name of the region for progress display
        subdirectory: Subdirectory name where CSV files are stored
        weekly_data: Dictionary to accumulate weekly data
    """
    for region_id in tqdm(region_list, desc=f"Processing {region_name} regions"):
        try:
            # Load the weekly CSV data
            weather_df = pd.read_csv(
                f"{DATA_DIR}/csvs/{subdirectory}/{region_id}_regional_weekly.csv",
                index_col=False,
            )
            # Extract data for each week and parameter
            for param in WEATHER_PARAMS:
                for week in range(1, 53):  # weeks 1-52
                    col_name = f"{param}_{week}"
                    if col_name in weather_df.columns:
                        # Get all values for this parameter and week across all locations/years
                        values = weather_df[col_name].dropna().values
                        weekly_data[week][param].extend(values.tolist())

        except Exception as e:
            print(f"Error processing {region_id}: {e}")
            continue


def compute_weekly_scalers():
    """
    Compute mean and std per week per feature for USA and Central America data.
    Uses existing processed weekly datasets to avoid memory issues.
    """
    print("Computing weekly scalers for USA and Central America data...")

    # Initialize storage for all weeks and features
    weekly_data = {
        week: {param: [] for param in WEATHER_PARAMS} for week in range(1, 53)
    }

    # Process USA and Central America regions
    process_regions(USA_REGIONS, "USA", "usa", weekly_data)
    process_regions(
        CENTRAL_AMERICA_REGIONS, "Central America", "centralamerica", weekly_data
    )
    print("Computing statistics per week per feature...")

    # Compute mean and std for each week and parameter
    weekly_scalers = {}

    for week in tqdm(range(1, 53), desc="Computing weekly statistics"):
        weekly_scalers[f"week_{week}"] = {"param_means": {}, "param_stds": {}}

        for param in WEATHER_PARAMS:
            if weekly_data[week][param]:  # Check if we have data
                values = np.array(weekly_data[week][param])
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))

                weekly_scalers[f"week_{week}"]["param_means"][param] = mean_val
                weekly_scalers[f"week_{week}"]["param_stds"][param] = std_val
            else:
                print(f"Warning: No data for Week {week}, {param}")
                # Use global scalers as fallback if available
                weekly_scalers[f"week_{week}"]["param_means"][param] = 0.0
                weekly_scalers[f"week_{week}"]["param_stds"][param] = 1.0

    # Save the weekly scalers
    output_path = f"{DATA_DIR}/processed/weekly_weather_param_scalers.json"
    with open(output_path, "w") as f:
        json.dump(weekly_scalers, f, indent=4)

    print(f"Weekly scalers saved to {output_path}")
    return weekly_scalers


if __name__ == "__main__":
    weekly_scalers = compute_weekly_scalers()
    print("Weekly scalers computation completed!")
