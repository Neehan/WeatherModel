import numpy as np
import pandas as pd

from src.crop_yield.dataloader.yield_dataloader import (
    CropDataset,
    split_train_test_by_year,
)


class NumpyCropDataset:
    """Dataset that returns flattened numpy arrays for traditional ML models"""

    def __init__(self, pytorch_dataset: CropDataset):
        self.X = []
        self.y = []

        for sample in pytorch_dataset.data:
            (
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                practices,
                soil,
                y_past,
                target_yield,
            ) = sample

            # Flatten all features into a single vector
            weather_flat = padded_weather.numpy().flatten()

            # Coords: use once and normalize (lat/360, lng/180)
            coord_flat = coord_processed.numpy()
            coord_flat = np.array([coord_flat[0] / 360, coord_flat[1] / 180])

            # Year: extract one value per year (not per week) and normalize
            year_flat = year_expanded.numpy()[::52]  # Take first week of each year
            year_flat = np.floor(year_flat)  # Remove fractional part
            year_flat = (year_flat - 1970) / 100.0  # Normalize

            soil_flat = soil.flatten()
            y_past_flat = y_past.flatten()

            # Concatenate all features (skip practices)
            features = np.concatenate(
                [
                    weather_flat,
                    coord_flat,
                    year_flat,
                    soil_flat,
                    y_past_flat,
                ]
            )

            self.X.append(features)
            self.y.append(target_yield[0])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def get_data(self):
        return self.X, self.y


def get_numpy_train_test_data(
    crop_df: pd.DataFrame,
    n_train_years: int,
    test_year: int,
    n_past_years: int,
    crop_type: str,
    country: str,
    test_gap: int = 0,
):

    # Use existing pytorch dataloader to create datasets
    train_dataset, test_dataset = split_train_test_by_year(
        crop_df,
        n_train_years,
        test_year,
        standardize=True,
        n_past_years=n_past_years,
        crop_type=crop_type,
        country=country,
        test_gap=test_gap,
    )

    # Convert to numpy datasets
    numpy_train = NumpyCropDataset(train_dataset)
    numpy_test = NumpyCropDataset(test_dataset)

    X_train, y_train = numpy_train.get_data()
    X_test, y_test = numpy_test.get_data()

    return (X_train, y_train), (X_test, y_test)
