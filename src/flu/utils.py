import pandas as pd
import logging
from constants import *


def filter_and_save_weather_data(
    regions,
    weather_csv_path="data/nasa_power/csvs/usa_weather_weekly.csv",
    save_path="data/flu_cases/weather_weekly.csv",
):
    logging.info("reading weather dataset")
    weather_df = pd.read_csv(weather_csv_path, index_col=[0]).drop(columns=["region"])

    # we only have cases data since 2010
    weather_df = weather_df.loc[weather_df.Year >= 2010.0]

    # filter out the coords for the region
    logging.info("filtering out coords for the region.")

    avg_weather_dfs = []

    for region_name in regions.keys():
        # for each city/region, get weather from the nearby locs and take mean
        avg_weather_df = (
            pd.concat(
                [
                    weather_df.loc[(weather_df.lat == lat) & (weather_df.lng == lng)]
                    for lat, lng in regions[region_name]
                ]
            )
            .groupby("Year")
            .agg("mean")
        )
        # make region the 4th column
        avg_weather_df.insert(4, "region", "")
        avg_weather_df["region"] = region_name

        avg_weather_dfs.append(avg_weather_df)

    weather_df = (
        # concat the averaged weathers for each region
        pd.concat(avg_weather_dfs).drop(columns=["Unnamed: 0"])
    )

    logging.info("saving the dataset")
    weather_df.to_csv(
        save_path,
        index=True,
    )