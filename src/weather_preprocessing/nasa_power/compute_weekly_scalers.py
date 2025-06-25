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


def compute_weekly_scalers():
    """
    Compute mean and std per week per feature for USA data only.
    Uses existing processed weekly datasets to avoid memory issues.
    """
    print("Computing weekly scalers for USA data...")

    # Initialize storage for all weeks and features
    weekly_data = {
        week: {param: [] for param in WEATHER_PARAMS} for week in range(1, 53)
    }

    # Process each USA region
    for region_id in tqdm(USA_REGIONS, desc="Processing USA regions"):
        try:
            print(f"Loading data for {region_id}")

            # Load the weekly CSV data
            weather_df = pd.read_csv(
                f"{DATA_DIR}/csvs/usa/{region_id}_regional_weekly.csv",
                index_col=False,
            )

            print(f"Processing {len(weather_df)} rows for {region_id}")

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

                print(
                    f"Week {week}, {param}: mean={mean_val:.4f}, std={std_val:.4f}, n_samples={len(values)}"
                )
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
