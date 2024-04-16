import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def read_and_concatenate_csv(file_paths):
    """
    Read multiple CSV files and concatenate them into a single DataFrame.
    """
    weather_dfs = [pd.read_csv(file, index_col=False) for file in file_paths]
    weather_df = pd.concat(weather_dfs)

    us_counties = pd.read_csv("data/uscounties.csv")[
        ["county", "state_name", "lat", "lng"]
    ]

    us_counties[["lat_radians", "lng_radians"]] = np.radians(
        us_counties[["lat", "lng"]]
    )
    weather_df[["lat_radians", "lng_radians"]] = np.radians(weather_df[["lat", "lng"]])

    # Create a BallTree using the latitude and longitude of the US counties
    tree = BallTree(
        us_counties[["lat_radians", "lng_radians"]].values, metric="haversine"
    )

    # Query the tree for the nearest county for each row in weather_df
    distances, indices = tree.query(
        weather_df[["lat_radians", "lng_radians"]].values, k=1
    )

    # Map the indices to county names
    weather_df["County"] = us_counties.iloc[indices.flatten()]["county"].values

    for col in ["State", "County"]:
        weather_df[col] = weather_df[col].str.lower()

    weather_df.drop_duplicates(subset={"State", "County", "Year"}, inplace=True)
    weather_df.rename(columns={"Year": "year"}, inplace=True)
    return weather_df


def add_state_county_khaki():
    us_counties = pd.read_csv("data/uscounties.csv")[
        ["county", "state_name", "lat", "lng"]
    ]
    for col in ["county", "state_name"]:
        us_counties[col] = us_counties[col].str.lower()
    us_counties.rename(
        columns={"county": "County", "state_name": "State"}, inplace=True
    )

    loc_ids = pd.read_csv("data/khaki_soybeans/Soybeans_Loc_ID.csv")
    data = pd.read_csv(
        "data/khaki_soybeans/soybean_data_soilgrid250_modified_states_9.csv"
    )

    us_counties = pd.merge(us_counties, loc_ids, on=["State", "County"])

    data = pd.merge(us_counties, data, on=["loc_ID"])
    return data


DATA_DIR = "data/nasa_power"
CORN_BELT = [
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
    # neighbors
    # "West Virginia",
    # "Virginia",
    # "North Carolina",
    # "Tennessee",
    # "Arkansas",
    # "Oklahoma",
    # "Colorado",
    # "Wyoming",
    # "Montana",
    # "Pennsylvania",
]

if __name__ == "__main__":
    input_paths = [f"{DATA_DIR}/{state}_regional_weekly.csv" for state in CORN_BELT]
    weather_df = read_and_concatenate_csv(input_paths)
    engineered_weather_cols = [
        f"{param}_{i}" for i in range(1, 53) for param in ["VAP", "VPD", "ET0"]
    ]
    weather_df = weather_df[["State", "County", "year"] + engineered_weather_cols]

    khaki_data = add_state_county_khaki()

    print(weather_df.head())
    khaki_data = pd.merge(
        weather_df, khaki_data, on=["State", "County", "year"], how="right"
    ).bfill()

    print("here's khaki:")
    print(khaki_data.head())

    khaki_data.to_csv(
        "data/khaki_soybeans/soybean_data_soilgrid250_modified_states_9_processed.csv",
        index=False,
    )
