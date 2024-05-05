import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging
import grids

# if set true, will average values over week
FREQUENCY = "daily"  # daily, weekly, monthly
ENGINEERED_FEATURES = True
DATA_DIR = "data/nasa_power"
REGION = "USA"


def ea_from_t2m(x):
    # teten's equation
    A = 17.27 if x > 0 else 21.87
    B = 237.3 if x > 0 else 265.5
    return 0.6108 * np.exp((A * x) / (x + B))


def compute_ET0(df):
    # https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation#FAO_56_Penman-Monteith_equation
    # Constants
    gamma = 0.066  # Psychrometric constant, kPa/C
    delta = (4098 * (0.6108 * np.exp(17.27 * df["T2M"] / (df["T2M"] + 237.3)))) / (
        df["T2M"] + 237.3
    ) ** 2  # Slope of vapor pressure curve
    rn = df["ALLSKY_SFC_SW_DWN"]  # Net irradiance
    G = 0  # Ground heat flux, usually 0 for daily computations
    et0 = (
        0.408 * delta * (rn - G)
        + gamma * (900 / (df["T2M"] + 273)) * df["WS2M"] * df["VPD"]
    ) / (delta + gamma * (1 + 0.34 * df["WS2M"]))
    return et0


def add_engineered_features(weather_df):
    weather_df = weather_df.copy()
    weather_df["VAP"] = weather_df["T2M"].apply(ea_from_t2m)

    # convert g / kg to kg/kg
    weather_df["QV2M"] /= 1000

    # source: https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    ea_actual = weather_df["QV2M"] / 1000 * 101.3 / (0.622 + 0.378 * weather_df["QV2M"])
    weather_df["VPD"] = weather_df["VAP"] - ea_actual  # vapor pressure difference

    # evapotranspiration for sample crop
    weather_df["ET0"] = compute_ET0(weather_df)
    return weather_df


def read_and_consolidate_data(region_name, part2=False):
    suffix = "_pt2" if part2 else ""
    with open(DATA_DIR + "/" + region_name + f"_data{suffix}.json", "r") as f:
        weather_json = json.load(f)
        f.close()
    weather_df = []

    print(f"processing weather data for {region_name} part {int(part2) + 1}.")
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
    if part2:
        weather_df.drop(columns=["alt"], inplace=True)
    return weather_df


def preprocess_weather_data(region_name):
    weather_df1 = read_and_consolidate_data(region_name)
    weather_df2 = read_and_consolidate_data(region_name, part2=True)

    weather_df = pd.merge(weather_df1, weather_df2, on=["lat", "lng", "Date"])
    # Convert the index to datetime format
    weather_df["Date"] = pd.to_datetime(weather_df.Date, format="%Y%m%d")
    weather_df["Year"] = weather_df["Date"].dt.year
    if FREQUENCY == "weekly":
        pivot_column = "Week"
        last_suffix = "_53"
        weather_df["Week"] = weather_df.Date.dt.isocalendar().week
    elif FREQUENCY == "monthly":
        pivot_column = "Month"
        last_suffix = "_13"
        weather_df["Month"] = weather_df.Date.dt.month
    else:  # daily frequency by default
        pivot_column = "doy"
        last_suffix = "_366"
        weather_df["doy"] = weather_df["Date"].dt.dayofyear.astype(int)

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

    # add the engineered features
    if ENGINEERED_FEATURES:
        weather_df = add_engineered_features(weather_df)

    weather_df = (
        weather_df.groupby(index_cols + [pivot_column]).agg("mean").reset_index()
    )
    weather_df = weather_df.pivot(index=index_cols, columns=pivot_column)
    weather_df.columns = [
        f"{measurement}_{int(i)}" for measurement, i in weather_df.columns
    ]
    weather_df.reset_index(drop=False, inplace=True)

    last_cols = [col for col in weather_df.columns if col.endswith(last_suffix)]
    weather_df.drop(columns=last_cols, inplace=True)
    weather_df = weather_df.bfill()
    weather_df = weather_df.fillna(method="pad", axis=1)
    weather_df["region"] = region_name

    weather_df.reset_index(inplace=True, drop=True)
    # reorder the columns
    non_weather_cols = ["region", "Year", "lat", "lng", "alt"]
    weather_df = weather_df[
        non_weather_cols
        + [col for col in weather_df.columns if col not in non_weather_cols]
    ]
    suffix = FREQUENCY

    weather_df.to_csv(f"{DATA_DIR}/{region_name}_regional_{suffix}.csv")
    print("total coords: ", len(weather_df) / 39)
    return weather_df


if __name__ == "__main__":
    regions = [f"{REGION.lower()}_{i}" for i in range(len(grids.GRID[REGION]))]
    for region in regions:
        _ = preprocess_weather_data(region)
