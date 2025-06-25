import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Configuration constants
WEATHER_BASE_PATH = "data/CropNet/WRF-HRRR Computed Dataset/data"
CROP_BASE_PATH = "data/CropNet/USDA Crop Dataset"
OUTPUT_DIR = "data/CropNet"
OUTPUT_FILE = "combined_cropnet_data.csv"

WEATHER_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
CROP_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
CROP_TYPES = ["Cotton", "Corn", "Soybeans", "WinterWheat"]

# Target states with weather data
TARGET_STATES = ["MISSISSIPPI", "LOUISIANA", "IOWA", "ILLINOIS"]

WEATHER_COLUMNS = {
    "Avg Temperature (K)": "temp_avg",
    "Max Temperature (K)": "temp_max",
    "Min Temperature (K)": "temp_min",
    "Precipitation (kg m**-2)": "precipitation",
    "Relative Humidity (%)": "humidity",
    "Wind Speed (m s**-1)": "wind_speed",
    "Downward Shortwave Radiation Flux (W m**-2)": "radiation",
    "Vapor Pressure Deficit (kPa)": "vpd",
}

WEATHER_COLUMN_ORDER = [
    "temp_avg",
    "temp_max",
    "temp_min",
    "precipitation",
    "humidity",
    "wind_speed",
    "radiation",
    "vpd",
]


def get_week_number(day_of_year):
    """Convert day of year to week number (1-52), ignoring leap year"""
    week = min(52, ((day_of_year - 1) // 7) + 1)
    return int(week)


def read_csv_files_for_path(path_pattern, description="files"):
    """Read and combine multiple CSV files from a path pattern"""
    csv_files = glob.glob(path_pattern)

    if not csv_files:
        return pd.DataFrame()

    monthly_data = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            monthly_data.append(df)
        except Exception as e:
            print(f"    Error reading {csv_file}: {e}")
            continue

    if not monthly_data:
        return pd.DataFrame()

    return pd.concat(monthly_data, ignore_index=True)


def filter_daily_weather_data(year_data):
    """Filter weather data to daily records only"""
    daily_data = year_data[year_data["Daily/Monthly"] == "Daily"].copy()

    if daily_data.empty:
        return daily_data

    # Calculate day of year and week
    daily_data["day_of_year"] = pd.to_datetime(
        daily_data[["Year", "Month", "Day"]]
    ).dt.dayofyear
    daily_data["week"] = daily_data["day_of_year"].apply(get_week_number)

    return daily_data


def calculate_weekly_weather_averages(daily_data, year):
    """Calculate weekly weather averages for each county"""
    weekly_data = []
    for fips, group in daily_data.groupby("FIPS Code"):
        county_weekly = create_county_weekly_record(group, year)
        weekly_data.append(county_weekly)

    return weekly_data


def create_county_weekly_record(group, year):
    """Create weekly weather record for a single county"""
    first_row = group.iloc[0]
    county_weekly = {
        "year": year,
        "state": first_row["State"],
        "county": first_row["County"],
        "fips": first_row["FIPS Code"],
        "lat": first_row["Lat (llcrnr)"],
        "lon": first_row["Lon (llcrnr)"],
    }

    # Check for missing weeks
    available_weeks = set(group["week"].unique())
    missing_weeks = set(range(1, 53)) - available_weeks
    if missing_weeks and len(missing_weeks) > 10:  # Only warn if many weeks missing
        print(f"      County {first_row['County']} missing {len(missing_weeks)} weeks")

    # Calculate weekly averages for each weather variable
    for week in range(1, 53):  # weeks 1-52
        week_data = group[group["week"] == week]

        if week_data.empty:
            # Fill missing weeks with NaN
            for orig_col, new_col in WEATHER_COLUMNS.items():
                county_weekly[f"{new_col}_{week}"] = np.nan
        else:
            # Calculate weekly values
            for orig_col, new_col in WEATHER_COLUMNS.items():
                if orig_col in week_data.columns:
                    if orig_col == "Precipitation (kg m**-2)":
                        # Sum precipitation instead of average
                        county_weekly[f"{new_col}_{week}"] = week_data[orig_col].sum()
                    else:
                        county_weekly[f"{new_col}_{week}"] = week_data[orig_col].mean()
                else:
                    county_weekly[f"{new_col}_{week}"] = np.nan

    return county_weekly


def process_weather_for_state_year(state_path, state_dir, year):
    """Process weather data for a single state and year"""
    print(f"  Processing {state_dir} {year}...")

    # Read and combine all monthly files for this state/year
    csv_pattern = os.path.join(state_path, "*.csv")
    year_data = read_csv_files_for_path(csv_pattern, "CSV files")

    if year_data.empty:
        print(f"    No data for {state_dir} {year}")
        return []

    # Filter for daily data and calculate weeks
    daily_data = filter_daily_weather_data(year_data)

    if daily_data.empty:
        print(f"    No daily data for {state_dir} {year}")
        return []

    # Calculate weekly averages
    weekly_data = calculate_weekly_weather_averages(daily_data, year)
    print(f"    Created {len(weekly_data)} county records")

    return weekly_data


def process_weather_files():
    """Process all weather files and create weekly averages"""
    print("Processing weather data...")

    weather_data = []

    # Process each year
    for year in WEATHER_YEARS:
        year_path = os.path.join(WEATHER_BASE_PATH, str(year))
        if not os.path.exists(year_path):
            print(f"  Year path not found: {year_path}")
            continue

        print(f"Processing weather year {year}...")

        # Get state directories
        state_dirs = [
            d
            for d in os.listdir(year_path)
            if os.path.isdir(os.path.join(year_path, d))
        ]
        print(f"  Found {len(state_dirs)} states: {state_dirs}")

        # Process each state
        for state_dir in state_dirs:
            state_path = os.path.join(year_path, state_dir)
            state_weather_data = process_weather_for_state_year(
                state_path, state_dir, year
            )
            weather_data.extend(state_weather_data)

    return create_weather_dataframe(weather_data)


def create_weather_dataframe(weather_data):
    """Create and validate weather DataFrame"""
    if not weather_data:
        print("No weather data found")
        return pd.DataFrame()

    weather_df = pd.DataFrame(weather_data)
    print(f"Total weather data processed: {len(weather_df)} records")
    print(f"Weather data columns: {len(weather_df.columns)}")

    validate_weather_data_completeness(weather_df)

    print(
        f"Sample weather record:\n{weather_df.iloc[0] if len(weather_df) > 0 else 'No data'}"
    )
    return weather_df


def validate_weather_data_completeness(weather_df):
    """Validate weather data for missing values"""
    weather_cols = [
        col
        for col in weather_df.columns
        if any(col.startswith(f"{wc}_") for wc in WEATHER_COLUMN_ORDER)
    ]

    missing_counts = weather_df[weather_cols].isna().sum()
    total_missing = missing_counts.sum()

    if total_missing > 0:
        print(f"WARNING: Found {total_missing} missing weather values")
        print_missing_values_by_week(weather_df, weather_cols)
    else:
        print("✓ No missing weather values found")


def print_missing_values_by_week(weather_df, weather_cols):
    """Print missing weather values grouped by week"""
    print("Missing values by week:")
    for week in range(1, 53):
        week_cols = [col for col in weather_cols if col.endswith(f"_{week}")]
        if week_cols:
            week_missing = weather_df[week_cols].isna().sum().sum()
            if week_missing > 0:
                print(f"  Week {week}: {week_missing} missing values")


def read_crop_csv_file(csv_file, crop):
    """Read and process a single crop CSV file"""
    try:
        df = pd.read_csv(csv_file)

        # Check for yield column - Cotton uses different units
        if crop.lower() == "cotton":
            yield_col = "YIELD, MEASURED IN LB / ACRE"
        elif crop.lower() == "winterwheat":
            yield_col = "YIELD, MEASURED IN BU / ACRE"
        else:
            yield_col = "YIELD, MEASURED IN BU / ACRE"

        if yield_col not in df.columns:
            print(f"      No yield column '{yield_col}' found in {csv_file}")
            return {}

        return process_crop_records(df, crop, yield_col)

    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")
        return {}


def process_crop_records(df, crop, yield_col):
    """Process individual crop records from DataFrame"""
    crop_data = {}

    # Filter for target states only
    df_filtered = df[df["state_name"].str.upper().isin(TARGET_STATES)]

    record_count = 0
    for _, row in df_filtered.iterrows():
        # Create unique key for location and year
        key = (
            row["year"],
            row["state_name"].upper(),
            row["county_name"].upper(),
        )

        # Initialize record if not exists
        if key not in crop_data:
            crop_data[key] = {
                "year": row["year"],
                "state": row["state_name"].upper(),
                "county": row["county_name"].upper(),
                "state_ansi": row["state_ansi"],
                "county_ansi": row["county_ansi"],
            }

        # Add crop yield
        crop_col = f"{crop.lower().replace('winter', 'winter ')}_yield"
        yield_value = row[yield_col] if pd.notna(row[yield_col]) else np.nan
        crop_data[key][crop_col] = yield_value

        if pd.notna(yield_value):
            record_count += 1

    return crop_data


def process_crop_for_year(crop_path, crop, year):
    """Process crop data for a single crop and year"""
    year_path = os.path.join(crop_path, str(year))
    if not os.path.exists(year_path):
        return {}

    # Find CSV files for this crop/year
    csv_files = glob.glob(os.path.join(year_path, "*.csv"))

    all_crop_data = {}
    for csv_file in csv_files:
        crop_data = read_crop_csv_file(csv_file, crop)
        all_crop_data.update(crop_data)

    return all_crop_data


def process_crop_files():
    """Process all crop yield files and create separate columns for each crop"""
    print("Processing crop data...")

    all_crop_data = {}

    for crop in CROP_TYPES:
        crop_path = os.path.join(CROP_BASE_PATH, crop)
        if not os.path.exists(crop_path):
            print(f"  Crop path not found: {crop_path}")
            continue

        print(f"Processing crop {crop}...")
        crop_records_before = len(all_crop_data)

        # Process each year
        for year in CROP_YEARS:
            year_crop_data = process_crop_for_year(crop_path, crop, year)

            # Debug: Check for key conflicts before merging
            conflicts = 0
            new_records = 0
            for key, record in year_crop_data.items():
                if key in all_crop_data:
                    # Merge the record
                    all_crop_data[key].update(record)
                    conflicts += 1
                else:
                    all_crop_data[key] = record
                    new_records += 1

            print(
                f"    Year {year}: {len(year_crop_data)} records, {new_records} new, {conflicts} merged"
            )

        crop_records_after = len(all_crop_data)
        print(
            f"  {crop} total contribution: {crop_records_after - crop_records_before} records"
        )

    return create_crop_dataframe(all_crop_data)


def create_crop_dataframe(all_crop_data):
    """Create and validate crop DataFrame"""
    if not all_crop_data:
        print("No crop data found")
        return pd.DataFrame()

    crop_df = pd.DataFrame(list(all_crop_data.values()))

    # Ensure all crop columns exist (fill with NaN if missing)
    for crop in CROP_TYPES:
        crop_col = f"{crop.lower().replace('winter', 'winter ')}_yield"
        if crop_col not in crop_df.columns:
            crop_df[crop_col] = np.nan

    print(f"Total crop data processed: {len(crop_df)} records")
    print(f"Crop data columns: {list(crop_df.columns)}")
    print(f"Sample crop record:\n{crop_df.iloc[0] if len(crop_df) > 0 else 'No data'}")

    print_crop_availability_summary(crop_df)

    return crop_df


def print_crop_availability_summary(crop_df):
    """Print summary of crop data availability"""
    for crop in CROP_TYPES:
        crop_col = f"{crop.lower().replace('winter', 'winter ')}_yield"
        if crop_col in crop_df.columns:
            non_null_count = crop_df[crop_col].notna().sum()
            print(f"  {crop_col}: {non_null_count} records")


def create_fips_code(state_ansi, county_ansi):
    """Create FIPS code from state and county ANSI codes"""
    if pd.isna(state_ansi) or pd.isna(county_ansi):
        return np.nan
    return int(state_ansi) * 1000 + int(county_ansi)


def prepare_data_for_merge(weather_df, crop_df):
    """Prepare weather and crop data for merging"""
    if crop_df.empty:
        return weather_df, crop_df

    # Create FIPS codes for crop data (weather already has them)
    crop_df["fips"] = crop_df.apply(
        lambda row: create_fips_code(row["state_ansi"], row["county_ansi"]), axis=1
    )

    return weather_df, crop_df


def print_merge_debug_info(weather_df, crop_df):
    """Print debug information for data merging"""
    print(f"Weather data before merge: {len(weather_df)} records")
    print(f"Crop data before merge: {len(crop_df)} records")

    print("Sample weather keys:")
    for i in range(min(5, len(weather_df))):
        row = weather_df.iloc[i]
        print(f"  {row['year']}, FIPS: {row['fips']}, {row['state']}, {row['county']}")

    print("Sample crop keys:")
    for i in range(min(5, len(crop_df))):
        row = crop_df.iloc[i]
        print(f"  {row['year']}, FIPS: {row['fips']}, {row['state']}, {row['county']}")


def perform_data_merge(weather_df, crop_df):
    """Perform the actual data merge"""
    # Use FIPS code and year for merge - more reliable than state/county names
    merge_keys = ["year", "fips"]

    merged_df = pd.merge(
        weather_df,
        crop_df.drop(columns=["state", "county"]),  # Drop original to avoid conflicts
        on=merge_keys,
        how="left",
        suffixes=("", "_crop"),
    )

    print(f"After merge: {len(merged_df)} records")
    return merged_df


def clean_merged_data(merged_df):
    """Clean up merged DataFrame"""
    # Drop duplicate/temporary columns
    cols_to_drop = [col for col in merged_df.columns if col.endswith("_crop")]
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)

    # Remove any duplicate columns
    if len(merged_df.columns) != len(set(merged_df.columns)):
        print("Warning: Found duplicate columns, removing duplicates...")
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    return merged_df


def print_merge_results(merged_df):
    """Print results of data merge"""
    print(f"Final combined data: {len(merged_df)} records")
    print(f"Combined data columns: {len(merged_df.columns)}")

    # Check how many records have crop data
    crop_cols = [col for col in merged_df.columns if col.endswith("_yield")]
    for crop_col in crop_cols:
        if crop_col in merged_df.columns:
            non_null_count = merged_df[crop_col].notna().sum()
            print(f"  {crop_col}: {non_null_count} records with data")


def combine_data(weather_df, crop_df):
    """Combine weather and crop data"""
    print("Combining weather and crop data...")

    if weather_df.empty:
        print("Cannot combine: no weather data")
        return pd.DataFrame()

    if crop_df.empty:
        print("Cannot combine: no crop data")
        return weather_df  # Return weather data only

    # Prepare data for merge
    weather_df, crop_df = prepare_data_for_merge(weather_df, crop_df)

    # Print debug info
    print_merge_debug_info(weather_df, crop_df)

    # Perform merge
    merged_df = perform_data_merge(weather_df, crop_df)

    # Clean up merged data
    merged_df = clean_merged_data(merged_df)

    # Print results
    print_merge_results(merged_df)

    return merged_df


def get_ordered_columns(final_df):
    """Get properly ordered columns for final DataFrame"""
    id_cols = ["year", "state", "county", "fips", "lat", "lon"]
    crop_cols = [col for col in final_df.columns if col.endswith("_yield")]
    weather_cols = [col for col in final_df.columns if col not in id_cols + crop_cols]

    # Sort weather columns by week number
    weather_cols_sorted = []
    for base_col in WEATHER_COLUMN_ORDER:
        for week in range(1, 53):
            col_name = f"{base_col}_{week}"
            if col_name in weather_cols:
                weather_cols_sorted.append(col_name)

    final_columns = id_cols + crop_cols + weather_cols_sorted

    # Only include existing columns and remove duplicates
    existing_columns = [col for col in final_columns if col in final_df.columns]

    # Remove duplicates while preserving order
    seen = set()
    unique_columns = []
    for col in existing_columns:
        if col not in seen:
            unique_columns.append(col)
            seen.add(col)

    # Add any remaining columns that weren't in our planned order
    for col in final_df.columns:
        if col not in seen:
            unique_columns.append(col)
            seen.add(col)

    return unique_columns


def save_final_data(final_df):
    """Save final DataFrame to CSV"""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reorder columns for better readability
    unique_columns = get_ordered_columns(final_df)
    final_df = final_df[unique_columns]

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    final_df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns[:15])}...")  # Show first 15 columns


def filter_records_with_yields(final_df):
    """Filter out records where all crop yields are missing"""
    print("Filtering records to keep only those with at least one crop yield...")

    crop_cols = [col for col in final_df.columns if col.endswith("_yield")]

    # Count records before filtering
    records_before = len(final_df)

    # Keep records where at least one crop yield is not null
    final_df_filtered = final_df[final_df[crop_cols].notna().any(axis=1)]

    records_after = len(final_df_filtered)
    records_removed = records_before - records_after

    print(f"Records before filtering: {records_before}")
    print(f"Records after filtering: {records_after}")
    print(f"Records removed (no crop yields): {records_removed}")

    return final_df_filtered


def print_final_summary(final_df):
    """Print final summary statistics"""
    print("\nSummary:")
    print(f"Years covered: {sorted(final_df['year'].unique())}")
    print(f"States covered: {len(final_df['state'].unique())}")
    print(f"Counties covered: {len(final_df.groupby(['state', 'county']).size())}")

    # Show crop yield availability
    crop_cols = [col for col in final_df.columns if col.endswith("_yield")]
    for crop_col in crop_cols:
        if crop_col in final_df.columns:
            non_null_count = final_df[crop_col].notna().sum()
            print(f"{crop_col}: {non_null_count} records")


def main():
    """Main function to process and combine all data"""
    print("Starting CropNet data processing...")

    # Process weather data
    weather_df = process_weather_files()

    # Process crop data
    crop_df = process_crop_files()

    # Combine the data
    final_df = combine_data(weather_df, crop_df)

    if not final_df.empty:
        # Filter records with yields
        final_df_filtered = filter_records_with_yields(final_df)

        # Save final data
        save_final_data(final_df_filtered)

        # Print summary
        print_final_summary(final_df_filtered)
    else:
        print("No data to save")


if __name__ == "__main__":
    main()
