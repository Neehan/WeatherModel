import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging


# if set true, will average values over week
WEEKLY = True
DATA_DIR = "data/nasa_power"


def preprocess_weather_data(state_name):
    print(f"preprocessing data for {state_name}.")
    with open(DATA_DIR + "/" + state_name + "_data.json", "r") as f:
        weather_json = json.load(f)
        f.close()
    weather_df = []

    logging.info(f"processing weather data for {state_name}")
    for chunk in tqdm(weather_json):
        for record in chunk["features"]:
            df = pd.DataFrame.from_dict(record["properties"]["parameter"]).reset_index(
                names="Date"
            )
            df["lng"] = record["geometry"]["coordinates"][0]
            df["lat"] = record["geometry"]["coordinates"][1]
            df["alt"] = record["geometry"]["coordinates"][2]
            weather_df.append(df)

    weather_df = pd.concat(weather_df, ignore_index=True)
    # print(weather_df.head())

    # Convert the index to datetime format
    weather_df["Date"] = pd.to_datetime(weather_df.Date, format="%Y%m%d")
    weather_df["Year"] = weather_df["Date"].dt.year
    if WEEKLY:
        pivot_column = "Week"
        last_suffix = "_53"
        weather_df["Week"] = weather_df.Date.dt.isocalendar().week

    else:
        pivot_column = "doy"
        last_suffix = "_366"
        weather_df["doy"] = weather_df["Date"].dt.dayofyear

    # weather_df.set_index(["lat", "lng", "Year"], inplace=True)
    weather_df.drop(columns=["Date"], inplace=True)

    index_cols = ["Year", "lat", "lng", "alt"]
    for col in weather_df.columns:
        if col not in index_cols:
            # -999 is the value for missing entries. If there are missing entries, make them null and
            # fill them later
            if len(weather_df.loc[weather_df[col] < -997.0, col]) > 0:
                print(f"{col} has missing values.")
            weather_df.loc[weather_df[col] < -997.0, col] = None

    weather_df = (
        weather_df.groupby(index_cols + [pivot_column]).agg("mean").reset_index()
    )
    weather_df = weather_df.pivot(index=index_cols, columns=pivot_column)
    weather_df.columns = [f"{measurement}_{i}" for measurement, i in weather_df.columns]
    weather_df.reset_index(drop=False, inplace=True)

    last_cols = [col for col in weather_df.columns if col.endswith(last_suffix)]
    weather_df.drop(columns=last_cols, inplace=True)
    weather_df = weather_df.bfill()
    weather_df = weather_df.fillna(method="pad", axis=1)
    weather_df["State"] = state_name

    weather_df.reset_index(inplace=True, drop=True)
    # reorder the columns
    non_weather_cols = ["State", "Year", "lat", "lng", "alt"]
    weather_df = weather_df[
        non_weather_cols
        + [col for col in weather_df.columns if col not in non_weather_cols]
    ]
    suffix = "weekly" if WEEKLY else "daily"

    weather_df.to_csv(f"{DATA_DIR}/{state_name}_regional_{suffix}.csv")
    print("total coords: ", len(weather_df) / 39)
    return weather_df


corn_belt = [
    # "Illinois",
    # "Indiana",
    # "Iowa",
    # "Kansas",
    # "Kentucky",
    # "Michigan",
    # "Minnesota",
    # "Missouri",
    # "Nebraska",
    # "North Dakota",
    # "Ohio",
    # "South Dakota",
    # "Wisconsin",
    # # neighbors
    # "West Virginia",
    # "Virginia",
    # "North Carolina",
    "Tennessee",
    "Arkansas",
    "Oklahoma",
    "Colorado",
    "Wyoming",
    "Montana",
    "Pennsylvania",
]

if __name__ == "__main__":
    for state in corn_belt:
        _ = preprocess_weather_data(state)
