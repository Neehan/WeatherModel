#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import requests
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, help="Country (mexico, brazil)")
    parser.add_argument(
        "--crop", required=True, help="Crop (corn, beans, sugarcane, wheat)"
    )
    args = parser.parse_args()

    # Data paths
    data_path = f"data/khaki_soybeans/khaki_{args.country}_multi_crop.csv"
    gadm_dir = Path("data/gadm")
    gadm_dir.mkdir(parents=True, exist_ok=True)

    country_codes = {
        "mexico": "MEX",
        "brazil": "BRA",
        "argentina": "ARG",
        "colombia": "COL",
        "peru": "PER",
        "chile": "CHL",
        "bolivia": "BOL",
        "ecuador": "ECU",
        "paraguay": "PRY",
        "uruguay": "URY",
        "venezuela": "VEN",
        "usa": "USA",
    }
    country_code = country_codes[args.country]
    gadm_file = gadm_dir / f"gadm41_{country_code}_2.json"

    # Load crop data
    df = pd.read_csv(data_path)
    crop_col = f"{args.crop}_yield"
    crop_counties = df[df[crop_col].notna()][["State", "County"]].drop_duplicates()
    # Remove spaces from state and county names to match GADM format
    crop_counties["State"] = crop_counties["State"].str.replace(" ", "")
    crop_counties["County"] = crop_counties["County"].str.replace(" ", "")
    print(f"Found {len(crop_counties)} counties with {args.crop} data")

    # Download/load GADM data
    if not gadm_file.exists():
        print(f"Downloading {country_code} admin boundaries...")
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_2.json"
        response = requests.get(url)
        gadm_file.write_bytes(response.content)
        print(f"Saved to {gadm_file}")

    print(f"Loading boundaries from {gadm_file}")
    gdf = gpd.read_file(gadm_file)
    print(f"Loaded {len(gdf)} admin units")

    # Filter to only states that have >=1 counties with crop data
    state_counts = crop_counties["State"].value_counts()
    relevant_states = state_counts[state_counts >= 1].index
    print(
        f"States with >=1 counties: {len(relevant_states)} out of {len(state_counts)}"
    )

    state_mask = gdf["NAME_1"].isin(relevant_states)
    gdf = gdf[state_mask].copy()
    print(f"Filtered to {len(gdf)} admin units in relevant states")

    # Match counties
    gdf["has_crop"] = False
    for _, crop_row in crop_counties.iterrows():
        state = crop_row["State"]
        county = crop_row["County"]
        mask = (gdf["NAME_1"].str.contains(state, case=False, na=False)) & (
            gdf["NAME_2"].str.contains(county, case=False, na=False)
        )
        gdf.loc[mask, "has_crop"] = True

    crop_count = gdf["has_crop"].sum()
    print(f"Matched {crop_count} admin units with {args.crop} data")

    # Plot
    fig, ax = plt.subplots(figsize=(15, 12))

    # Gray for no crop
    gdf[gdf["has_crop"] == False].plot(
        ax=ax, color="lightgray", edgecolor="white", linewidth=0.3
    )

    # Green for crop
    gdf[gdf["has_crop"] == True].plot(
        ax=ax, color="green", edgecolor="white", linewidth=0.3
    )

    # Add grid every 5 degrees
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Set ticks every 5 degrees
    lon_ticks = np.arange(
        np.floor(bounds[0] / 5) * 5, np.ceil(bounds[2] / 5) * 5 + 1, 5
    )
    lat_ticks = np.arange(
        np.floor(bounds[1] / 5) * 5, np.ceil(bounds[3] / 5) * 5 + 1, 5
    )

    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xlim(bounds[0], bounds[2])  # Set x limits to data bounds
    ax.set_ylim(bounds[1], bounds[3])  # Set y limits to data bounds
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)

    # Add legend with bigger text
    legend_elements = [
        Patch(facecolor="green", label="Available"),
        Patch(facecolor="lightgray", label="Not available"),
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc="best")

    # Save
    Path("notebooks/img").mkdir(parents=True, exist_ok=True)
    output_file = f"notebooks/img/{args.country}_{args.crop}_admin2.pdf"
    plt.savefig(output_file, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
