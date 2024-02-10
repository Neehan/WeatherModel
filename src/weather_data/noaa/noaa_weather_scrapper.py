import requests
import json
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

SAVE_DIR = "data/noaa"
DATASET_ID = "GHCND"
LIMIT = 1000  # Number of records per request
API_KEY = os.environ.get(
    "NOAA_API_KEY", "NOT_FOUND"
)  # Replace with your actual API key


NOAA_DATATYPES = {
    "AWND": "Average Wind Speed",
    "WSF1": "Fastest 2-minute wind speed",
    "WSF2": "Fastest 1-minute wind speed",
    "WSFG": "Peak gust wind speed",
    "WDF1": "Direction of Fastest 2-minute wind speed",
    "WDF2": "Direction of Fastest 1-minute wind speed",
    "WDFG": "Direction of Peak gust wind speed",
    "SNOW": "Snowfall",
    "SNWD": "Snow Depth",
    "WESF": "Water equivalent of snowfall",
    "WT01": "Fog, ice fog, or freezing fog (may include heavy fog)",
    "WT03": "Thunder",
    "WT05": "Hail (may include small hail)",
    "WT07": "Dust, volcanic ash, blowing dust, blowing sand, or blowing obstruction",
    "WT08": "Smoke or haze",
    "WT11": "High or damaging winds",
    "TAVG": "Average Temperature.",
    "TMIN": "Minimum Temperature.",
    "TMAX": "Maximum Temperature.",
    "TSUN": "Total sunshine for the period",
    "PRCP": "Precipitation",
    "ACSH": "Average cloudiness sunrise to sunset from manual observations",
    "SN52": "Minimum soil temperature with sod cover at 10 cm depth",
    "SX52": "Maximum soil temperature with sod cover at 10 cm depth",
}


def make_request(url, headers, params, max_retries=3):
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    response = session.get(url, headers=headers, params=params)
    return response


def fetch_data_for_state(state_id, start_date, end_date, output_dir):
    current_start_date = start_date
    temp_files = []

    while current_start_date < end_date:
        current_end_date = min(current_start_date + timedelta(days=6), end_date)
        offset = 1
        total_records = None
        first_iteration = True

        while total_records is None or offset <= total_records:
            params = {
                "datasetid": DATASET_ID,
                "datatypeid": ",".join(NOAA_DATATYPES.keys()),
                "locationid": state_id,
                "startdate": current_start_date.strftime("%Y-%m-%d"),
                "enddate": current_end_date.strftime("%Y-%m-%d"),
                "limit": LIMIT,
                "offset": offset,
            }
            headers = {"token": API_KEY}
            response = make_request(
                f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data", headers, params
            )

            if response and response.status_code == 200:
                data = response.json()
                if first_iteration:
                    total_records = data["metadata"]["resultset"]["count"]
                    pbar = tqdm(
                        total=total_records, desc=f"Fetching {state_id}", leave=False
                    )
                    first_iteration = False

                received_records = len(data.get("results", []))
                pbar.update(received_records)

                # Save each chunk to a temporary file
                temp_file_path = os.path.join(
                    output_dir, f"temp_{state_id}_{offset}.json"
                )
                with open(temp_file_path, "w") as temp_file:
                    json.dump(data["results"], temp_file)
                temp_files.append(temp_file_path)

                offset += LIMIT
            else:
                break

        current_start_date = current_end_date + timedelta(days=1)
        if not first_iteration:
            pbar.close()

    return temp_files


def fetch_data(states, start_date, end_date, output_dir=SAVE_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_temp_files = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for state in states:
            futures.append(
                executor.submit(
                    fetch_data_for_state, state, start_date, end_date, output_dir
                )
            )

        for future in tqdm(
            as_completed(futures), total=len(states), desc="Fetching States"
        ):
            temp_files = future.result()
            all_temp_files.extend(temp_files)

    return all_temp_files


def combine_json_files(file_paths, output_file):
    combined_data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            combined_data.extend(json.load(f))
        os.remove(file_path)

    with open(output_file, "w") as f:
        json.dump(combined_data, f)


# Example usage
states = ["FIPS:17"]  # Example state ID
start_date = datetime(2010, 1, 1)
end_date = datetime(2010, 1, 7)

temp_files = fetch_data(states, start_date, end_date)
output_file = os.path.join(SAVE_DIR, "combined_weather_data.json")
combine_json_files(temp_files, output_file)
print(f"Combined data saved in {output_file}")
