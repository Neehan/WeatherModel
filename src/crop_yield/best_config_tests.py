#!/usr/bin/env python3
"""
Best Configuration Tests

This script loads the best configuration from grid search results and runs a single test.
Results are saved to data/best_config_tests/ as TSV files
"""

import os
import logging
import pandas as pd
import time
from typing import Dict, Tuple, Optional, List

from src.crop_yield.yield_main import main as yield_main_func
from src.utils.utils import setup_logging, get_model_params

# Setup logging
setup_logging(rank=0)
logger = logging.getLogger(__name__)


def is_baseline_model(model: str) -> bool:
    """Check if model is a baseline model (no pretrained variant)"""
    return model.lower() in ["xgboost", "randomforest"]


def get_grid_search_file_path(
    model: str, crop_type: str, country: str, grid_search_results_dir: str
) -> str:
    """Get the grid search results file path for the given model, crop type, and country"""
    results_dir = os.path.join(grid_search_results_dir, model, "extreme_years")

    if not os.path.exists(results_dir):
        raise FileNotFoundError(
            f"Grid search results directory not found: {results_dir}"
        )

    # Baseline models use different file naming (includes test_type suffix)
    if is_baseline_model(model):
        filename = f"baseline_grid_search_{model}_{crop_type}_{country}_extreme.tsv"
    else:
        filename = f"grid_search_{model}_pretrained_{crop_type}_{country}.tsv"

    file_path = os.path.join(results_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Grid search results file not found: {file_path}")

    logger.info(f"Using grid search results file: {file_path}")
    return file_path


def load_grid_search_results(file_path: str) -> pd.DataFrame:
    """Load grid search results from TSV file"""
    try:
        df = pd.read_csv(file_path, sep="\t")
        logger.info(f"Loaded grid search results with {len(df)} experiments")
        return df
    except Exception as e:
        raise Exception(f"Failed to load grid search results from {file_path}: {e}")


def find_best_config(df: pd.DataFrame, model: str) -> Dict:
    """Find the best configuration based on R² score"""
    if df.empty:
        raise ValueError("No grid search results found")

    def extract_r2_mean(r2_str):
        """Extract mean R² value from 'X.XXX ± Y.YYY' format"""
        if pd.isna(r2_str) or r2_str == "FAILED":
            return float("-inf")
        try:
            return float(r2_str.split(" ± ")[0])
        except:
            return float("-inf")

    # Baseline models use "r2" column, regular models use "year_15_r2" column
    if is_baseline_model(model):
        r2_col = "r2"
    else:
        r2_col = "year_15_r2"

    if r2_col not in df.columns:
        raise ValueError(f"Expected column {r2_col} not found in results")

    valid_df = df[df[r2_col] != "FAILED"].copy()

    if valid_df.empty:
        raise ValueError("No successful experiments found in grid search results")

    valid_df["r2_mean"] = valid_df[r2_col].apply(extract_r2_mean)  # type: ignore
    best_idx = valid_df["r2_mean"].idxmax()  # type: ignore
    best_row = valid_df.loc[best_idx]

    # Build config based on model type
    if is_baseline_model(model):
        best_config = {
            "model": best_row["model"],
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": (
                int(best_row["max_depth"]) if pd.notna(best_row["max_depth"]) else None
            ),
            "learning_rate": (
                float(best_row["learning_rate"])
                if "learning_rate" in best_row and pd.notna(best_row["learning_rate"])
                else None
            ),
            "r2_score": best_row["r2_mean"],
        }

        logger.info(f"Best configuration found:")
        logger.info(f"  Model: {best_config['model']}")
        logger.info(f"  n_estimators: {best_config['n_estimators']}")
        logger.info(f"  max_depth: {best_config['max_depth']}")
        logger.info(f"  Learning rate: {best_config['learning_rate']}")
        logger.info(f"  R² score: {best_config['r2_score']:.4f}")
    else:
        best_config = {
            "model": best_row["model"],
            "method": best_row["method"],
            "beta": best_row["beta"],
            "batch_size": int(best_row["batch_size"]),
            "init_lr": best_row["init_lr"],
            "r2_score": best_row["r2_mean"],
        }

        logger.info(f"Best configuration found:")
        logger.info(f"  Model: {best_config['model']}")
        logger.info(f"  Method: {best_config['method']}")
        logger.info(f"  Beta: {best_config['beta']}")
        logger.info(f"  Batch size: {best_config['batch_size']}")
        logger.info(f"  Learning rate: {best_config['init_lr']}")
        logger.info(f"  R² score: {best_config['r2_score']:.4f}")

    return best_config


def create_test_config(
    model: str,
    crop_type: str,
    country: str,
    test_type: str,
    n_train_years: int,
    best_config: Dict,
) -> Dict:
    """Create configuration for the test"""
    config = best_config.copy()

    # Set test parameters
    if test_type == "overall":
        description = f"Overall test with {n_train_years} years of history"
    elif test_type == "ahead_pred":
        description = f"Ahead prediction test with {n_train_years} years of history"
    else:
        raise ValueError(
            f"Invalid test_type: {test_type}. Must be 'overall' or 'ahead_pred'"
        )

    config.update(
        {
            "test_type": test_type,
            "n_train_years": n_train_years,
            "crop_type": crop_type,
            "country": country,
            "description": description,
        }
    )

    # Set common parameters
    n_past_years = 6 if country != "mexico" else 4
    config.update(
        {
            "n_past_years": n_past_years,
            "seed": 1234,
            "test_year": None,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
        }
    )

    # Baseline models don't need neural network specific parameters
    if not is_baseline_model(model):
        # Set model size parameters
        config["model_size_params"] = get_model_params("small")

        # Set pretrained model path
        pretrained_paths = {
            "weatherformer": "data/trained_models/pretraining/weatherformer_1.9m_latest.pth",
            "weatherautoencoder": "data/trained_models/pretraining/weatherautoencoder_1.9m_latest.pth",
            "weatherformersinusoid": "data/trained_models/pretraining/weatherformer_sinusoid_2.0m_latest.pth",
            "weatherformermixture": "data/trained_models/pretraining/weatherformer_mixture_2.0m_latest.pth",
            "weatherautoencodermixture": "data/trained_models/pretraining/weatherautoencoder_2.0m_latest.pth",
            "weatherautoencodersinusoid": "data/trained_models/pretraining/weatherautoencoder_2.0m_latest.pth",
            "simmtm": "data/trained_models/pretraining/simmtm_1.9m_latest.pth",
            "chronos": "data/trained_models/pretraining/weatherautoencoder_1.9m_latest.pth",
            "linear": None,
        }
        config["pretrained_model_path"] = pretrained_paths.get(config["model"], None)

        # Set additional neural network parameters
        config.update(
            {
                "n_epochs": 40,
                "decay_factor": None,
                "n_warmup_epochs": 10,
                "model_size": "small",
                "use_optimal_lr": False,
                "n_mixture_components": 1,
                "resume_from_checkpoint": None,
            }
        )

    return config


def run_test(
    config: Dict,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[List[float]],
]:
    """Run the test with given configuration"""
    logger.info(f"Starting test: {config['test_type']} - {config['description']}")
    logger.info(f"  Test type: {config['test_type']}")
    logger.info(f"  Training years: {config['n_train_years']}")

    try:
        start_time = time.time()
        avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values = yield_main_func(config)
        end_time = time.time()
        runtime_seconds = end_time - start_time

        logger.info(
            f"Completed test: RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}, R² = {avg_r2:.3f} ± {std_r2:.3f}"
        )
        logger.info(f"Runtime: {runtime_seconds:.1f} seconds")

        return avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values

    except Exception as e:
        logger.error(f"Failed test: {str(e)}", exc_info=True)
        return None, None, None, None, None


def save_single_result(
    model: str,
    crop_type: str,
    country: str,
    test_type: str,
    config: Dict,
    avg_rmse: Optional[float],
    std_rmse: Optional[float],
    avg_r2: Optional[float],
    std_r2: Optional[float],
):
    """Save a single test result immediately to TSV file (append mode for HPC safety)"""
    output_dir = "data/best_config_tests"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f"best_config_tests_{model}_{crop_type}_{country}_{test_type}.tsv"
    )

    # Format results like grid search with ± notation
    rmse_str = f"{avg_rmse:.3f} ± {std_rmse:.3f}" if avg_rmse is not None else "FAILED"
    r2_str = f"{avg_r2:.3f} ± {std_r2:.3f}" if avg_r2 is not None else "FAILED"

    row = {
        "model": model,
        "crop_type": crop_type,
        "country": country,
        "test_type": test_type,
        "n_train_years": config["n_train_years"],
        "rmse": rmse_str,
        "r2": r2_str,
    }

    # Check if file exists to determine if we need header
    file_exists = os.path.exists(output_file)

    # Create DataFrame and append to file
    df = pd.DataFrame([row])

    if file_exists:
        # Append without header
        df.to_csv(output_file, sep="\t", index=False, mode="a", header=False)
        logger.info(f"Appended result to: {output_file}")
    else:
        # Create new file with header
        df.to_csv(output_file, sep="\t", index=False, mode="w", header=True)
        logger.info(f"Created new results file: {output_file}")

    logger.info(
        f"Saved result: {test_type} {config['n_train_years']}y - RMSE: {rmse_str}, R²: {r2_str}"
    )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run best configuration test")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--crop-type", required=True, help="Crop type")
    parser.add_argument("--country", default="usa", help="Country")
    parser.add_argument(
        "--grid-search-results-dir",
        default="data/results",
        help="Grid search results directory",
    )
    parser.add_argument(
        "--test-type",
        choices=["overall", "ahead_pred"],
        required=True,
        help="Test type to run",
    )

    args = parser.parse_args()

    # Load grid search results and find best config
    file_path = get_grid_search_file_path(
        args.model, args.crop_type, args.country, args.grid_search_results_dir
    )
    df = load_grid_search_results(file_path)
    best_config = find_best_config(df, args.model)

    # Determine years to test based on test type
    if args.test_type == "overall":
        years_to_test = [15, 30]
    else:  # ahead_pred
        years_to_test = [15]

    # Run tests for each year configuration
    for n_train_years in years_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {args.test_type} test with {n_train_years} years")
        logger.info(f"{'='*60}")

        # Create test configuration
        config = create_test_config(
            args.model,
            args.crop_type,
            args.country,
            args.test_type,
            n_train_years,
            best_config,
        )

        # Run test
        avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values = run_test(config)

        # Save result immediately (HPC-safe)
        save_single_result(
            args.model,
            args.crop_type,
            args.country,
            args.test_type,
            config,
            avg_rmse,
            std_rmse,
            avg_r2,
            std_r2,
        )

        # Print summary
        if avg_rmse is not None:
            logger.info(
                f"Test completed and saved ({n_train_years}y): RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}, R² = {avg_r2:.3f} ± {std_r2:.3f}"
            )
        else:
            logger.info(f"Test FAILED and logged ({n_train_years}y)")

    logger.info(f"\nAll {args.test_type} tests completed and saved!")


if __name__ == "__main__":
    main()
