import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from region_weather_scrapper import WEATHER_PARAMS

np.random.seed(1234)
torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

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
    "West Virginia",
    "Virginia",
    "North Carolina",
    "Tennessee",
    "Arkansas",
    "Oklahoma",
    "Colorado",
    "Wyoming",
    "Montana",
    "Pennsylvania",
]


def read_and_concatenate_csv(file_paths):
    """
    Read multiple CSV files and concatenate them into a single DataFrame.
    """
    weather_dfs = [pd.read_csv(file, index_col=False) for file in file_paths]
    return pd.concat(weather_dfs, ignore_index=True)


def preprocess_data(weather_df):
    """
    Preprocess the DataFrame: drop duplicates and standardize columns.
    """
    # Drop duplicates based on year, latitude, and longitude
    weather_df = weather_df.drop_duplicates(subset=["Year", "lat", "lng"])

    # Standardize all columns except 'State'
    columns_to_standardize = weather_df.columns.difference(["State"])
    scaler = StandardScaler()
    weather_df.loc[:, columns_to_standardize] = scaler.fit_transform(
        weather_df[columns_to_standardize]
    )

    return weather_df


def split_data(weather_df):
    """
    Split the data into training, validation, and testing sets.
    """
    # Shuffle the dataset
    weather_df = weather_df.sample(frac=1).reset_index(drop=True)

    # Split the data 90% training and 10% validation
    train_df, val_df = train_test_split(weather_df, test_size=0.1)
    return train_df, val_df


def shuffle_measurements(data):
    """
    Shuffle data along the last dimension, consistently for all weeks within a batch,
    but potentially differently for each batch.
    """
    batch_size = data.shape[0]
    measurement_size = data.shape[-1]

    # Initialize an empty array with the same shape as the input data
    shuffled_data = np.empty_like(data)

    for i in tqdm(range(batch_size)):
        # Generate random indices for each batch
        indices = np.arange(measurement_size)
        np.random.shuffle(indices)

        # Apply the same shuffled indices to all weeks within this batch
        shuffled_data[i] = data[i, :, indices].T

    return shuffled_data


def create_datasets(train_df, val_df):
    """
    Create PyTorch DataLoaders for training, validation, and testing sets.
    """

    def create_dataset(weather_df):
        # Extracting weather measurements
        weather_measurements = weather_df[
            [
                f"{param}_{i}"
                for i in range(1, 53)
                for param in WEATHER_PARAMS.split(",")
            ]
        ].values

        weather_measurements = weather_measurements.reshape(
            -1, 52, len(WEATHER_PARAMS.split(","))
        )
        # shuffle weather measurements so that the model does not associate
        # a dimension with a measurement. This is important because we want to
        # use different measurements later on.
        weather_measurements = shuffle_measurements(weather_measurements)

        # Extracting year, coordinates
        year = weather_df[["Year"]].values
        coords = weather_df[["lat", "lng"]].values

        # Converting to tensors
        weather_tensor = torch.tensor(weather_measurements, dtype=torch.float32)
        year_tensor = torch.tensor(year, dtype=torch.float32)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        print("weather tensor shape: ", weather_tensor.shape)
        print("year tensor shape: ", year_tensor.shape)
        print("coords tensor shape: ", coords_tensor.shape)

        return TensorDataset(weather_tensor, year_tensor, coords_tensor)

    # Creating datasets
    print("creating training dataset")
    train_dataset = create_dataset(train_df)
    print("creating validation dataset")
    val_dataset = create_dataset(val_df)

    return train_dataset, val_dataset


def save_datasets(train_dataset, val_dataset, file_path):
    """
    Save the DataLoaders for future use.
    """
    torch.save(train_dataset, file_path + "/train_dataset.pth")
    torch.save(val_dataset, file_path + "/val_dataset.pth")


if __name__ == "__main__":
    input_paths = [f"{DATA_DIR}/{state}_regional_weekly.csv" for state in CORN_BELT]

    print("reading weather datasets")
    weather_df = read_and_concatenate_csv(input_paths)
    print("preprocessing.")
    weather_df = preprocess_data(weather_df)

    train_df, val_df = split_data(weather_df)
    print("creating dataloaders.")
    train_dataset, val_dataset = create_datasets(train_df, val_df)
    print("saving dataloaders.")
    save_datasets(train_dataset, val_dataset, DATA_DIR)
