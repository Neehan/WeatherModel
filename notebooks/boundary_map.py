#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', required=True, help='Country (mexico, brazil)')
    parser.add_argument('--crop', required=True, help='Crop (corn, beans, sugarcane, wheat)')
    args = parser.parse_args()

    # Data paths
    data_path = f'data/khaki_soybeans/khaki_{args.country}_multi_crop.csv'
    gadm_dir = Path('data/gadm')
    gadm_dir.mkdir(parents=True, exist_ok=True)

    country_codes = {
        'mexico': 'MEX',
        'brazil': 'BRA',
        'argentina': 'ARG',
        'colombia': 'COL',
        'peru': 'PER',
        'chile': 'CHL',
        'bolivia': 'BOL',
        'ecuador': 'ECU',
        'paraguay': 'PRY',
        'uruguay': 'URY',
        'venezuela': 'VEN',
        'usa': 'USA'
    }
    country_code = country_codes[args.country]
    gadm_file = gadm_dir / f'gadm41_{country_code}_2.json'

    # Load crop data
    df = pd.read_csv(data_path)
    crop_col = f'{args.crop}_yield'
    crop_counties = df[df[crop_col].notna()][['State', 'County']].drop_duplicates()
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

    # Match counties
    gdf['has_crop'] = False
    for _, crop_row in crop_counties.iterrows():
        state = crop_row['State']
        county = crop_row['County']
        mask = (gdf['NAME_1'].str.contains(state, case=False, na=False)) & \
               (gdf['NAME_2'].str.contains(county, case=False, na=False))
        gdf.loc[mask, 'has_crop'] = True

    crop_count = gdf['has_crop'].sum()
    print(f"Matched {crop_count} admin units with {args.crop} data")

    # Plot
    fig, ax = plt.subplots(figsize=(15, 12))

    # Gray for no crop
    gdf[gdf['has_crop'] == False].plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)

    # Green for crop
    gdf[gdf['has_crop'] == True].plot(ax=ax, color='green', edgecolor='white', linewidth=0.3)

    ax.set_title(f'{args.country.title()} - {args.crop.title()} Admin 2 Coverage')
    ax.axis('off')

    # Save
    Path('notebooks/data/img').mkdir(parents=True, exist_ok=True)
    output_file = f'notebooks/data/img/{args.country}_{args.crop}_admin2.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()