import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.drought.constants import TQDM_OUTPUT
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE, DRY_RUN


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
        if test_dataset:  # test on missouri and kansas
            # test only final year
            self.index = data[
                (data["State"] == test_states[0]) | (data["State"] == test_states[1])
            ][["year", "loc_ID"]].reset_index(drop=True)
        else:
            self.index = data[
                (data["State"] != test_states[0]) & (data["State"] != test_states[1])
            ][["year", "loc_ID"]].reset_index(drop=True)

        self.data = []

        which_dataset = "Test" if test_dataset else "Training"

        for idx in range(1000 if DRY_RUN else len(self.index)):
            year, loc_ID = self.index.iloc[idx].values.astype("int")
            # look up last n years of data for a location
            query_data_true = data[(data["year"] <= year) & (data["loc_ID"] == loc_ID)]
            query_data = query_data_true
            # attention mask for the transformer model
            mask = np.zeros((n_past_years + 1,)).astype(bool)
            # pad shorter history with zeros
            if len(query_data) < n_past_years + 1:
                zero_df = pd.DataFrame(
                    0,
                    index=range(n_past_years + 1 - len(query_data)),
                    columns=query_data.columns,
                )
                query_data_true = pd.concat([zero_df, query_data_true]).reset_index(
                    drop=True
                )
                query_data = pd.concat([zero_df, query_data]).reset_index(drop=True)
                # make attention mask zero at the padded rows
                mask[: len(zero_df)] = True
            elif len(query_data) > n_past_years + 1:
                query_data_true = query_data_true.tail(n_past_years + 1)
                query_data = query_data.tail(n_past_years + 1)

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
            year = query_data["year"].values.astype("float32")
            year = (year - 1970.0) / 100.0
            coord = torch.FloatTensor(
                query_data[["lat", "lng"]].values.astype("float32")
            )

            # get the true yield
            y = query_data_true.iloc[-1:]["yield"].values.astype("float32").copy()
            y_past = query_data_true["yield"].values.astype("float32")
            # the current year's yield the target variable, so replace it with last year's yield
            y_past[-1] = y_past[-2]

            self.data.append(((weather, practices, soil, year, coord, y_past, mask), y))

    def __len__(self):
        return 1000 if DRY_RUN else len(self.index)

    def __getitem__(self, idx):
        return self.data[idx]  # weather, practices, soil, year, y, y_mean

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
    crop_df: pd.DataFrame, test_states, n_past_years: int, batch_size: int
):
    train_dataset, test_dataset = split_train_test_by_year(
        crop_df, test_states, True, n_past_years=n_past_years
    )

    train_loader = train_dataset.get_data_loader(batch_size=batch_size)
    test_loader = test_dataset.get_data_loader(batch_size=batch_size)

    return train_loader, test_loader
