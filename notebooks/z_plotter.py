import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import argparse
from typing import Tuple, Dict, List, Optional
import seaborn as sns
from tqdm import tqdm

# Add src to path
import sys

sys.path.append("../src")
import os

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# Comprehensive rcParams for publication-ready neurips paper - SET ONCE
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 24,
        "axes.titlesize": 26,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "lines.linewidth": 3,
        "lines.markersize": 12,
        "lines.markeredgewidth": 2.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "font.weight": "normal",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "grid.alpha": 0.3,
        "grid.linewidth": 1,
        "figure.figsize": (20, 6),
        "axes.grid": True,
    }
)

os.environ["DRY_RUN"] = "0"

from src.utils.constants import DATA_DIR, TOTAL_WEATHER_VARS
from src.crop_yield.dataloader.yield_dataloader import (
    get_train_test_loaders,
    read_usa_dataset,
    read_argentina_dataset,
)
from src.crop_yield.dataloader.cropnet_dataloader import (
    get_cropnet_train_test_loaders,
    read_cropnet_dataset,
)

# Import all possible model classes
from src.crop_yield.models.weatherformer_yield_model import WeatherFormerYieldModel
from src.crop_yield.models.weatherbert_yield_model import WeatherBERTYieldModel
from src.pretraining.models.weatherautoencoder import WeatherAutoencoder
from src.crop_yield.models.weatherformer_mixture_yield_model import (
    WeatherFormerMixtureYieldModel,
)
from src.crop_yield.models.weatherformer_sinusoid_yield_model import (
    WeatherFormerSinusoidYieldModel,
)
from src.crop_yield.models.weatherautoencoder_mixture_yield_model import (
    WeatherAutoencoderMixtureYieldModel,
)
from src.crop_yield.models.weatherautoencoder_sine_yield_model import (
    WeatherAutoencoderSineYieldModel,
)


def load_model(model_path: str, device: torch.device, n_past_years: int = 6):
    """Load model from checkpoint - create instance first, then load pretrained weights."""
    print(f"Loading model from: {model_path}")

    # Load the pretrained model
    pretrained_model = torch.load(model_path, map_location=device, weights_only=False)

    # Create fresh model instance with current constants
    if "weatherformer_sinusoid" in model_path:
        model = WeatherFormerSinusoidYieldModel(
            name="weatherformer_sinusoid_yield",
            device=device,
            k=1,  # Default k value
            weather_dim=TOTAL_WEATHER_VARS,  # Use current constant
            n_past_years=n_past_years,
            num_heads=10,
            num_layers=4,
            hidden_dim_factor=20,
        )
        # Load the pretrained weights
        model.load_pretrained(pretrained_model)
    elif "weatherformer" in model_path and "sinusoid" not in model_path:
        model = WeatherFormerYieldModel(
            name="weatherformer_yield",
            device=device,
            weather_dim=TOTAL_WEATHER_VARS,  # Use current constant
            n_past_years=n_past_years,
            num_heads=10,
            num_layers=4,
            hidden_dim_factor=20,
        )
        # Load the pretrained weights
        model.load_pretrained(pretrained_model)
    elif "weatherautoencoder" in model_path:
        model = WeatherAutoencoder(
            weather_dim=TOTAL_WEATHER_VARS,  # Use current constant
            output_dim=TOTAL_WEATHER_VARS,
            device=device,
            num_heads=10,
            num_layers=2,
            hidden_dim_factor=20,
        )
        # Load the pretrained weights
        model.load_pretrained(pretrained_model)
    else:
        # For other model types, use directly
        model = pretrained_model

    model.eval()
    return model


def extract_latents(
    model, dataloader, device: torch.device, max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent representations and years from the dataloader."""
    latents = []
    years = []
    locations = []

    print("Extracting latent representations...")
    is_bert = isinstance(model, WeatherAutoencoder)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
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
            ) = batch

            # Move to device
            padded_weather = padded_weather.to(device)
            coord_processed = coord_processed.to(device)
            year_expanded = year_expanded.to(device)
            interval = interval.to(device)
            weather_feature_mask = weather_feature_mask.to(device)
            y_past = y_past.to(device)

            if is_bert:
                model_outputs = model(
                    padded_weather,
                    coord_processed,
                    year_expanded,
                    interval,
                    weather_feature_mask,
                )
                z = model_outputs
            else:
                # Forward pass to get latents
                model_outputs = model(
                    padded_weather,
                    coord_processed,
                    year_expanded,
                    interval,
                    weather_feature_mask,
                    y_past,
                )
                z = model_outputs[1]

            z = z[:, -52:, :]
            # Get the year for this batch (use the first timestep of the last year)
            # year_expanded shape: (batch_size, n_years * seq_len)
            batch_size = padded_weather.shape[0]
            seq_len = z.shape[1]  # weekly data
            n_years = year_expanded.shape[1] // seq_len

            # Extract the year from the last year's first timestep for each sample
            for i in range(batch_size):
                # Get the year from the last year (most recent)
                year_val = int(
                    year_expanded[i, -seq_len].item()
                )  # Last year, first week
                years.append(year_val)

                # Extract latent for this sample
                # z is (batch_size, seq_len, n_features), mean pool over seq dimension
                # z_sample = z[i].mean(dim=0).cpu().numpy()
                z_sample = z[i].cpu().numpy().flatten()

                # Debug: check individual sample
                if batch_idx == 0 and i == 0:
                    print(
                        f"DEBUG - Sample z_sample mean: {z_sample.mean():.3f}, std: {z_sample.std():.3f}"
                    )
                    print(
                        f"DEBUG - Sample z_sample range: [{z_sample.min():.3f}, {z_sample.max():.3f}]"
                    )

                latents.append(z_sample)

                # Extract location (lat, lng)
                loc = coord_processed[i].cpu().numpy()
                locations.append(loc)

            # Only limit if max_samples is specified
            if max_samples and len(latents) > max_samples:
                print(f"Limiting to {max_samples} samples for visualization")
                break

    if not latents:
        raise ValueError("No latents extracted! Check if model outputs are compatible.")

    latents = np.array(latents)
    years = np.array(years)
    locations = np.array(locations)

    print(f"Extracted {len(latents)} latent representations")
    print(f"Latent dimension: {latents.shape[1]}")
    print(f"Years range: {years.min()} - {years.max()}")

    # Print samples per year
    unique_years, year_counts = np.unique(years, return_counts=True)
    print("Samples per year:")
    for year, count in zip(unique_years, year_counts):
        print(f"  {year}: {count} samples")

    return latents, years, locations


def plot_latents_pca(
    latents: np.ndarray,
    years: np.ndarray,
    locations: np.ndarray,
    save_path: Optional[str] = None,
    years_to_plot: Optional[List[int]] = None,
):
    """Apply PCA and plot latents colored by year."""
    print("Applying PCA...")

    # Filter data if specific years are requested
    if years_to_plot:
        mask = np.isin(years, years_to_plot)
        latents = latents[mask]
        years = years[mask]
        locations = locations[mask]
        print(f"Filtered to {len(latents)} samples for years: {years_to_plot}")

        # Print samples per filtered year
        unique_years, year_counts = np.unique(years, return_counts=True)
        print("Samples per filtered year:")
        for year, count in zip(unique_years, year_counts):
            print(f"  {year}: {count} samples")

        # Debug: Check raw latent statistics
    print(f"Raw latents - mean: {latents.mean():.3f}, std: {latents.std():.3f}")
    print(f"Raw latents - min: {latents.min():.3f}, max: {latents.max():.3f}")
    print(
        f"Raw latents - per dim std range: [{latents.std(axis=0).min():.3f}, {latents.std(axis=0).max():.3f}]"
    )

    # Since dimensions have similar scales, apply PCA directly without standardization
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"PCA output range: [{latents_2d.min():.3f}, {latents_2d.max():.3f}]")

    # Remove extreme left outliers to make plot more professional
    # Use 5th percentile to filter out extreme left values
    left_threshold = np.percentile(latents_2d[:, 0], 5)
    valid_mask = latents_2d[:, 0] > left_threshold

    latents_2d_filtered = latents_2d[valid_mask]
    years_filtered = years[valid_mask]
    locations_filtered = locations[valid_mask]

    print(f"Filtered out {(~valid_mask).sum()} outlier samples")
    print(f"Remaining samples: {len(latents_2d_filtered)}")

    # Create the plot using the paper style already set at top
    plt.figure()  # Use the figure.figsize from rcParams (20, 6)

    # Get unique years and create a color map using seaborn default colors
    unique_years = sorted(np.unique(years_filtered))
    colors = sns.color_palette(n_colors=len(unique_years))

    print(f"Plotting {len(unique_years)} unique years: {unique_years}")

    # Plot each year with semi-transparent colors
    for i, year in enumerate(unique_years):
        mask = years_filtered == year
        plt.scatter(
            latents_2d_filtered[mask, 0],
            latents_2d_filtered[mask, 1],
            c=[colors[i]],
            label=str(year),
            alpha=0.7,
            s=80,  # Bigger dots for better visibility (2x larger)
            edgecolors="white",
            linewidth=0.3,
        )

        # Set aspect ratio to make plot taller and improve proportions
    plt.gca().set_aspect(1.0)  # Make y-axis appear twice as tall relative to x-axis

    # Set fixed axis limits for consistent shape across all plots
    plt.xlim([-30, 35])
    plt.ylim([-15, 15])

    # Remove axes labels and ticks for clean neurips presentation
    plt.xticks([])
    plt.yticks([])

    # Clean, academic-style legend for NeurIPS paper - horizontal layout
    legend = plt.legend(
        loc="upper center",
        fontsize=36,  # Half the previous size
        frameon=False,  # Remove box around legend
        bbox_to_anchor=(0.5, -0.05),  # Center horizontally, below plot area
        handletextpad=0.05,  # Minimal padding between marker and text
        borderaxespad=0,
        borderpad=0,  # Remove internal padding
        markerscale=5.0,  # 2x larger legend markers
        ncol=len(unique_years),  # Display all years in one row
        columnspacing=0.2,  # Minimal space between columns
    )

    plt.grid(True, alpha=0.2)
    # Add bottom margin to accommodate the horizontal legend
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # plt.show()

    return latents_2d, pca


def main():
    parser = argparse.ArgumentParser(
        description="Plot latent representations from yield prediction model"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--crop-type",
        default="soybean",
        choices=["soybean", "corn", "wheat", "sunflower"],
    )

    parser.add_argument(
        "--n-past-years",
        default=1,
        type=int,
        help="Number of past years used in training",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="Batch size for data loading"
    )
    parser.add_argument(
        "--test-year", default=2012, type=int, help="Test year to analyze"
    )
    parser.add_argument(
        "--use-cropnet",
        action="store_true",
        help="Use CropNet dataset instead of soybean dataset",
    )
    parser.add_argument("--save-plot", help="Path to save the plot")
    parser.add_argument(
        "--years-to-plot",
        type=int,
        nargs="+",
        help="Specific years to plot (e.g., --years-to-plot 2016 2017 2018)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to extract for visualization",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device, args.n_past_years)

    # Load dataset and create dataloader
    if args.use_cropnet:
        crop_df = read_cropnet_dataset(DATA_DIR)
        train_loader, test_loader = get_cropnet_train_test_loaders(
            crop_df,
            args.crop_type,
            n_train_years=9,  # Use more years to get more data
            test_year=args.test_year,
            n_past_years=args.n_past_years,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        crop_df = read_usa_dataset(DATA_DIR)
        train_loader, test_loader = get_train_test_loaders(
            crop_df,
            n_train_years=9,  # Use more years to get more data
            test_year=args.test_year,
            n_past_years=args.n_past_years,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            crop_type=args.crop_type,
        )

    # Combine train and test loaders to get all data
    print("Processing training data...")
    train_latents, train_years, train_locations = extract_latents(
        model, train_loader, device, args.max_samples
    )

    print("Processing test data...")
    test_latents, test_years, test_locations = extract_latents(
        model, test_loader, device, args.max_samples
    )

    # Combine all data
    all_latents = np.concatenate([train_latents, test_latents], axis=0)
    all_years = np.concatenate([train_years, test_years], axis=0)
    all_locations = np.concatenate([train_locations, test_locations], axis=0)

    # Plot
    latents_2d, pca = plot_latents_pca(
        all_latents, all_years, all_locations, args.save_plot, args.years_to_plot
    )

    return latents_2d, all_years, all_locations, pca


if __name__ == "__main__":
    main()
