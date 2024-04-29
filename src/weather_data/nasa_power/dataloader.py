import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import json
from grids import GRID

MAX_LENGTH = 180
FREQUENCY = "daily"
FREQUENCY_DAYS = 1
SEQUENCE_LEN = 365

# FREQUENCY = "weekly"
# FREQUENCY_DAYS = 7
# SEQUENCE_LEN = 52

# FREQUENCY = "monthly"
# FREQUENCY_DAYS = 30
# SEQUENCE_LEN = 12

FILE_SUFFIX = FREQUENCY
NUM_YEARS = 39

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

WEATHER_PARAMS = list(WEATHER_PARAMS.values()) + ["ET0", "VAP", "VPD"]

np.random.seed(1234)
torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

DATA_DIR = "data/nasa_power"
REGION = "SOUTHAMERICA"
region_coords = GRID[REGION]
region_names = [f"{REGION.lower()}_{i}" for i in range(len(region_coords))]


def read_and_concatenate_csv(file_paths):
    """
    Read multiple CSV files and concatenate them into a single DataFrame.
    """
    weather_dfs = [pd.read_csv(file, index_col=False) for file in tqdm(file_paths)]
    return pd.concat(weather_dfs, ignore_index=True).drop_duplicates(
        subset=["Year", "lat", "lng"]
    )


def preprocess_data(weather_df):
    """
    Preprocess the DataFrame: drop duplicates and standardize columns.
    """
    print("Standardizing the columns.")
    if FREQUENCY == "weekly" and REGION == "USA":
        param_means = dict()
        param_stds = dict()
    else:
        with open(DATA_DIR + f"/processed/weather_param_scalers.json", "r") as f:
            scalers = json.load(f)
            param_means, param_stds = scalers["param_means"], scalers["param_stds"]
            f.close()

    for param in tqdm(WEATHER_PARAMS):
        weather_cols = [f"{param}_{i}" for i in range(1, SEQUENCE_LEN + 1)]
        if FREQUENCY == "weekly" and REGION == "USA":
            param_mean = weather_df[weather_cols].values.mean()
            param_std = weather_df[weather_cols].values.std()
            param_means[param] = param_mean
            param_stds[param] = param_std
        else:
            param_mean = param_means[param]
            param_std = param_stds[param]

        weather_df[weather_cols] = (weather_df[weather_cols] - param_mean) / param_std

    with open(DATA_DIR + f"/processed/weather_param_scalers.json", "w") as f:
        json.dump({"param_means": param_means, "param_stds": param_stds}, f)
        f.close()

    print("Sorting by location and year")
    weather_df = weather_df.sort_values(by=["lat", "lng", "Year"])
    return weather_df


def create_dataset(weather_df):
    seq_len = SEQUENCE_LEN
    weather_measurements = weather_df[
        [f"{param}_{i}" for i in range(1, seq_len + 1) for param in WEATHER_PARAMS]
    ].values.reshape(-1, NUM_YEARS * seq_len, len(WEATHER_PARAMS))

    # now reshape to have MAX_LENGTH
    num_segments = NUM_YEARS * seq_len // MAX_LENGTH
    weather_measurements = np.concatenate(
        (
            weather_measurements[:, : num_segments * MAX_LENGTH, :],
            weather_measurements[:, -MAX_LENGTH:, :],  # add the remaining data
        ),
        axis=1,
    )
    num_segments += 1  # we added one additional data point above
    weather_measurements = weather_measurements.reshape(
        -1, MAX_LENGTH, len(WEATHER_PARAMS)
    )

    coords = weather_df[["lat", "lng"]].values.reshape(-1, NUM_YEARS, 2)

    num_regions = coords.shape[0]

    coords = np.repeat(coords[:, :1, :], num_segments, axis=1).reshape(-1, 2)

    # we need an index
    index = np.repeat(np.arange(num_segments)[None, :, None], num_regions, axis=0)
    # daily, weekly, monthly etc
    data_frequency = FREQUENCY_DAYS * np.ones((num_regions, num_segments, 1))
    index = np.stack([index, data_frequency], axis=2).reshape(-1, 2)

    # Converting to tensors
    weather_tensor = torch.tensor(weather_measurements, dtype=torch.float32)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    index_tensor = torch.tensor(index, dtype=torch.float32)

    # print("weather tensor shape: ", weather_tensor.shape)
    # print("coords tensor shape: ", coords_tensor.shape)
    # print("index tensor shape: ", index_tensor.shape)

    return TensorDataset(weather_tensor, coords_tensor, index_tensor)


def save_dataset(train_dataset, part_id, file_path):
    """
    Save the DataLoaders for future use.
    """
    torch.save(
        train_dataset,
        file_path + f"/processed/weather_dataset_{FILE_SUFFIX}_{part_id}.pth",
    )


if __name__ == "__main__":
    num_regions = len(region_names)
    input_paths = [
        f"{DATA_DIR}/{region}_regional_{FILE_SUFFIX}.csv"
        for region in region_names[2 * (num_regions // 3 + 1) :]
    ]
    weather_df = read_and_concatenate_csv(input_paths)
    print("saving the merged dataset")
    weather_df.to_csv(DATA_DIR + f"/{REGION.lower()}_weather_{FILE_SUFFIX}_pt3.csv")

    # print("reading weather datasets")
    # weather_df = pd.read_csv(
    #     DATA_DIR + f"/{REGION.lower()}_weather_{FILE_SUFFIX}_pt1.csv", index_col=[0]
    # )

    print("preprocessing.")
    weather_df = preprocess_data(weather_df)

    assert (
        len(weather_df) % NUM_YEARS == 0
    ), "dataset length is not divisible by number of years"
    dataset_length = len(weather_df)

    print("creating dataset.")
    chunk_length = 320 * NUM_YEARS
    for start_index in tqdm(range(0, dataset_length, chunk_length)):
        part_id = start_index // chunk_length + 36
        end_index = min(dataset_length, start_index + chunk_length)
        train_dataset = create_dataset(weather_df.iloc[start_index:end_index])
        save_dataset(train_dataset, part_id, DATA_DIR)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05])
