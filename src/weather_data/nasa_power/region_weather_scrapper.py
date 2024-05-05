import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import os
from grids import GRID

PART1 = True
REGION = "USA"  # "USA", "CENTRALAMERICA", "CANADA" or "SOUTHAMERICA"


WEATHER_PARAMS = {
    "Temperature at 2 Meters (C)": "T2M",
    "Temperature at 2 Meters Maximum (C)": "T2M_MAX",
    "Temperature at 2 Meters Minimum (C)": "T2M_MIN",
    "Wind Direction at 2 Meters (Degrees)": "WD2M",
    "Wind Speed at 2 Meters (m/s)": "WS2M",
    "Surface Pressure (kPa)": "PS",
    "Specific Humidity at 2 Meters (g/Kg)": "QV2M",
    "Precipitation Corrected (mm/day)": "PRECTOTCORR",
    "All Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)": "ALLSKY_SFC_SW_DWN",
    "Evapotranspiration Energy Flux (MJ/m^2/day)": "EVPTRNS",
    "Profile Soil (the layer from the surface down to the bedrock) Moisture (0 to 1)": "GWETPROF",
    "Snow Depth (cm)": "SNODP",
    "Dew/Frost Point at 2 Meters (C)": "T2MDEW",
    "Cloud Amount (%)": "CLOUD_AMT",
    # additional 14
    "Evaporation Land (kg/m^2/s * 10^6)": "EVLAND",
    "Wet Bulb Temperature at 2 Meters (C)": "T2MWET",
    "Land Snowcover Fraction (0 to 1)": "FRSNO",
    "All Sky Surface Longwave Downward Irradiance (MJ/m^2/day)": "ALLSKY_SFC_LW_DWN",
    "All Sky Surface PAR Total (MJ/m^2/day)": "ALLSKY_SFC_PAR_TOT",
    "All Sky Surface Albedo (0 to 1)": "ALLSKY_SRF_ALB",
    "Precipitable Water (cm)": "PW",
    "Surface Roughness (m)": "Z0M",
    "Surface Air Density (kg/m^3) ": "RHOA",
    "Relative Humidity at 2 Meters (%)": "RH2M",
    "Cooling Degree Days Above 18.3 C": "CDD18_3",
    "Heating Degree Days Below 18.3 C": "HDD18_3",
    "Total Column Ozone (Dobson units)": "TO3",
    "Aerosol Optical Depth 55": "AOD_55",
}

if PART1:
    WEATHER_PARAMS = dict(list(WEATHER_PARAMS.items())[:14])
else:
    WEATHER_PARAMS = dict(list(WEATHER_PARAMS.items())[14:])


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
    suffix = "" if PART1 else "_pt2"
    with open(f"{SAVE_DIR}/{state_name}_data{suffix}.json", "w") as file:
        json.dump(all_data, file)


def fetch_weather_for_state(
    state_name, latitude_min, latitude_max, longitude_min, longitude_max
):

    start_date = datetime.strptime("19840101", "%Y%m%d")
    end_date = datetime.strptime("20221231", "%Y%m%d")
    date_ranges = split_dates(start_date, end_date)

    chunk_counter = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
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


def get_coordinates(coord_map, region_name):

    region_id = int(region_name.split("_")[1])
    (latitude_max, longitude_min), (latitude_min, longitude_max) = coord_map[region_id]
    # print(latitude_min, latitude_max, longitude_min, longitude_max)
    return latitude_min, latitude_max, longitude_min, longitude_max


if __name__ == "__main__":
    region_coords = GRID[REGION]
    region_names = [f"{REGION.lower()}_{i}" for i in range(len(region_coords))]

    for i, region_name in enumerate(region_names):
        print(f"fetching weather for {region_name}.")
        latitude_min, latitude_max, longitude_min, longitude_max = get_coordinates(
            region_coords, region_name
        )
        print(f"latitude  range: {latitude_min:.2f}, {latitude_max:.2f}")
        print(f"longitude range: {longitude_min:.2f}, {longitude_max:.2f}")

        fetch_weather_for_state(
            region_name, latitude_min, latitude_max, longitude_min, longitude_max
        )
