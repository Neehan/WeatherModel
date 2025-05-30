import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.constants import DRY_RUN, MAX_CONTEXT_LENGTH, TOTAL_WEATHER_VARS


class CropDataset(Dataset):
    def __init__(self, data, test_states, test_dataset=False, n_past_years=5):
        self.weather_cols = [f"W_{i}_{j}" for i in range(1, 7) for j in range(1, 53)]
        self.practice_cols = [f"P_{i}" for i in range(1, 15)]
        soil_measurements = [
            "bdod",
            "cec",
            "cfvo",
            "clay",
            "nitrogen",
            "ocd",
            "ocs",
            "phh2o",
            "sand",
            "silt",
            "soc",
        ]
        soil_depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
        self.soil_cols = [
            f"{measure}_mean_{depth}"
            for measure in soil_measurements
            for depth in soil_depths
        ]

        # Define weather indices used in preprocessing
        self.weather_indices = torch.tensor([7, 8, 11, 1, 2, 29])

        if test_dataset:  # test on missouri and kansas
            # test only final year
            candidate_data = data[
                (data["State"] == test_states[0]) | (data["State"] == test_states[1])
            ]
        else:
            candidate_data = data[
                (data["State"] != test_states[0]) & (data["State"] != test_states[1])
            ]

        # Filter to only include cases where we have complete historical data
        valid_indices = []
        for _, row in candidate_data.iterrows():
            year, loc_ID = row["year"], row["loc_ID"]
            # Get the actual data we would use for this location/year
            historical_data = data[
                (data["year"] <= year) & (data["loc_ID"] == loc_ID)
            ].tail(n_past_years + 1)
            # Only include if we have exactly the right amount of data
            if len(historical_data) == n_past_years + 1:
                valid_indices.append((year, loc_ID))

        self.index = pd.DataFrame(valid_indices, columns=["year", "loc_ID"])
        self.data = []

        for idx in range(1000 if DRY_RUN else len(self.index)):
            year, loc_ID = self.index.iloc[idx].values.astype("int")
            # Get exactly n_past_years + 1 years of data for this location
            query_data = data[(data["year"] <= year) & (data["loc_ID"] == loc_ID)].tail(
                n_past_years + 1
            )

            weather = (
                query_data[self.weather_cols]
                .values.astype("float32")
                .reshape((-1, 6, 52))
            )  # 6 measurements, 52 weeks
            practices = (
                query_data[self.practice_cols]
                .values.astype("float32")
                .reshape((-1, 14))
            )  # 14 practices
            soil = (
                query_data[self.soil_cols].values.astype("float32").reshape((-1, 11, 6))
            )  # 11 measurements, at 6 depths
            year_data = query_data["year"].values.astype("float32")
            coord = torch.FloatTensor(
                query_data[["lat", "lng"]].values.astype("float32")
            )

            # get the true yield
            y = query_data.iloc[-1:]["yield"].values.astype("float32").copy()
            y_past = query_data["yield"].values.astype("float32")
            # the current year's yield is the target variable, so replace it with last year's yield
            y_past[-1] = y_past[-2]

            # Preprocess weather data for the model
            n_years, n_features, seq_len = weather.shape

            # Check context length constraint
            if n_years * seq_len > MAX_CONTEXT_LENGTH:
                raise ValueError(
                    f"n_years * seq_len = {n_years * seq_len} is greater than MAX_CONTEXT_LENGTH = {MAX_CONTEXT_LENGTH}"
                )

            # Transpose and reshape weather data: (n_years, n_features, seq_len) -> (n_years * seq_len, n_features)
            weather = weather.transpose(0, 2, 1)  # (n_years, seq_len, n_features)
            weather = weather.reshape(
                n_years * seq_len, n_features
            )  # (n_years * seq_len, n_features)

            # Process coordinates - use only the first coordinate (same for all years in this location)
            coord_processed = coord[0, :]  # (2,)

            # Expand year to match the sequence length
            # year_data is [n_years], need to repeat each year for seq_len timesteps
            year_expanded = (
                torch.FloatTensor(year_data).unsqueeze(1).expand(n_years, seq_len)
            )  # [n_years, seq_len]
            year_expanded = year_expanded.contiguous().view(
                n_years * seq_len
            )  # [n_years * seq_len]

            # Create padded weather with specific weather indices
            padded_weather = torch.zeros(
                (seq_len * n_years, TOTAL_WEATHER_VARS),
            )
            padded_weather[:, self.weather_indices] = torch.FloatTensor(weather)

            # Create weather feature mask
            weather_feature_mask = torch.ones(
                TOTAL_WEATHER_VARS,
                dtype=torch.bool,
            )
            weather_feature_mask[self.weather_indices] = False

            # Create temporal interval (weekly data)
            interval = torch.full((1,), 7, dtype=torch.float32)

            self.data.append(
                (
                    padded_weather,  # preprocessed weather data
                    coord_processed,  # processed coordinates
                    year_expanded,  # expanded year data
                    interval,  # temporal interval
                    weather_feature_mask,  # feature mask
                    practices,  # practices (unchanged)
                    soil,  # soil (unchanged)
                    y_past,  # past yields
                    y,  # target yield
                )
            )

    def __len__(self):
        return 1000 if DRY_RUN else len(self.index)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_loader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size, shuffle=False)


def split_train_test_by_year(
    soybean_df: pd.DataFrame,
    test_states,
    standardize: bool = True,
    n_past_years: int = 5,
):
    data = soybean_df[
        soybean_df["year"] > 1981.0
    ]  # must be > 1981 otherwise all past data is just 0

    if standardize:
        cols_to_standardize = [
            col
            for col in data.columns
            if col not in ["loc_ID", "year", "State", "County", "lat", "lng"]
        ]
        data = pd.merge(
            data[["year", "State", "loc_ID", "lat", "lng"]],
            (data[cols_to_standardize] - data[cols_to_standardize].mean())
            / data[cols_to_standardize].std(),
            left_index=True,
            right_index=True,
        )
    data = data.fillna(0)

    train_dataset = CropDataset(
        data.copy(), test_states, test_dataset=False, n_past_years=n_past_years
    )
    test_dataset = CropDataset(
        data.copy(), test_states, test_dataset=True, n_past_years=n_past_years
    )

    # Return the train and test datasets
    return train_dataset, test_dataset


def read_soybean_dataset(data_dir: str):
    full_filename = (
        "khaki_soybeans/soybean_data_soilgrid250_modified_states_9_processed.csv"
    )
    soybean_df = pd.read_csv(data_dir + full_filename)
    soybean_df["year_std"] = soybean_df["year"]
    soybean_df = soybean_df.sort_values(["loc_ID", "year"])
    return soybean_df


def get_train_test_loaders(
    crop_df: pd.DataFrame,
    test_states,
    n_past_years: int,
    batch_size: int,
    shuffle: bool,
):
    train_dataset, test_dataset = split_train_test_by_year(
        crop_df, test_states, standardize=True, n_past_years=n_past_years
    )

    train_loader = train_dataset.get_data_loader(batch_size=batch_size)
    test_loader = test_dataset.get_data_loader(batch_size=batch_size)

    return train_loader, test_loader
