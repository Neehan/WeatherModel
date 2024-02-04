import requests
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import os
import argparse

WEATHER_PARAMS = {
    "Temperature at 2 Meters (C)": "T2M",
    "Temperature at 2 Meters Maximum (C)": "T2M_MAX",
    "Temperature at 2 Meters Minimum (C)": "T2M_MIN",
    "Wind Direction at 2 Meters (Degrees)": "WD2M",
    "Wind Speed at 2 Meters (m/s)": "WS2M",
    "Profile Soil (the layer from the surface down to the bedrock) Moisture (0 to 1)": "GWETPROF",
    "Surface Pressure (kPa)": "PS",
    "Specific Humidity at 2 Meters (g/Kg)": "QV2M",
    "Snow Depth (cm)": "SNODP",
    "Precipitation Corrected (mm/day)": "PRECTOTCORR",
    "All Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)": "ALLSKY_SFC_SW_DWN",
    "Evapotranspiration Energy Flux (MJ/m^2/day)": "EVPTRNS",
    "Dew/Frost Point at 2 Meters (C)": "T2MDEW",
    "Cloud Amount (%)": "CLOUD_AMT",
}

WEATHER_PARAMS = ",".join(WEATHER_PARAMS.values())
SAVE_DIR = "data/nasa_power"


# Function to split dates into intervals of 366 days or less
def split_dates(start, end):
    date_ranges = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=90), end)
        date_ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return date_ranges


def fetch_data_from_api(params):
    endpoint = "https://power.larc.nasa.gov/api/temporal/daily/regional"
    retries = 3
    base_delay = 2  # Initial delay in seconds

    for attempt in range(retries):
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < retries - 1:  # Not the final retry
                wait_time = base_delay * (2**attempt)  # Exponential backoff
                print(
                    f"Error on attempt {attempt + 1}. Retrying in {wait_time} seconds..."
                )
                print(e)
                time.sleep(wait_time)
            else:
                print(e)
                # print(f"Error fetching data for params {params}: {e}")
                return None


def save_data_chunk(state_name, result, counter):
    # Save the data chunk into its own file
    file_name = f"{SAVE_DIR}/{state_name}_{counter}.json"
    with open(file_name, "w") as file:
        json.dump(result, file)
        file.close()


def consolidate_data(state_name, total_chunks):
    all_data = []

    # Read and consolidate all the data chunks
    for counter in range(total_chunks):
        file_name = f"{SAVE_DIR}/{state_name}_{counter}.json"
        with open(file_name, "r") as file:
            all_data.append(json.load(file))
    # Save consolidated data for the state
    with open(f"{SAVE_DIR}/{state_name}_data.json", "w") as file:
        json.dump(all_data, file)


def fetch_weather_for_state(state_name, coordinates):
    longs, lats = zip(*coordinates)
    latitude_min, latitude_max = min(lats), max(lats)
    longitude_min, longitude_max = min(longs), max(longs)

    start_date = datetime.strptime("19840101", "%Y%m%d")
    end_date = datetime.strptime("20221231", "%Y%m%d")
    date_ranges = split_dates(start_date, end_date)

    chunk_counter = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for start, end in date_ranges:
            params = {
                "latitude-min": latitude_min,
                "latitude-max": latitude_max,
                "longitude-min": longitude_min,
                "longitude-max": longitude_max,
                "parameters": WEATHER_PARAMS,
                "community": "AG",
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "format": "JSON",
            }
            futures.append(executor.submit(fetch_data_from_api, params))

        for future in tqdm(futures):
            result = future.result()
            if result:
                save_data_chunk(state_name, result, chunk_counter)
                chunk_counter += 1

    consolidate_data(state_name, chunk_counter)

    # Now that data is consolidated, remove individual chunks
    for counter in range(chunk_counter):
        file_name = f"{SAVE_DIR}/{state_name}_{counter}.json"
        os.remove(file_name)


corn_belt = [
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Michigan",
    "Minnesota",
    "Missouri",
    "Nebraska",
    "North Dakota",
    "Ohio",
    "South Dakota",
    "Wisconsin",
]

# corn_belt = ["Illinois"]


with open("state_coords.json", "r") as f:
    state_coords = json.load(f)
    f.close()

for state_name in corn_belt:
    print(f"fetching weather for {state_name}.")
    fetch_weather_for_state(state_name, state_coords[state_name])
