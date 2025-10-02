import os
import logging
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from src.crop_yield.dataloader.numpy_yield_dataloader import get_numpy_train_test_data
from src.crop_yield.dataloader.yield_dataloader import (
    read_usa_dataset,
    read_non_us_dataset,
)
from src.utils.constants import DATA_DIR, CROP_YIELD_STATS, EXTREME_YEARS, TEST_YEARS
from src.utils.utils import setup_logging


class BaselineGridSearch:
    """Grid search for baseline models (XGBoost, Random Forest)"""

    def __init__(
        self,
        model: str,
        crop_type: str,
        output_dir: str,
        country: str,
        test_type: str = "extreme",
    ):
        self.model = model
        self.crop_type = crop_type
        self.output_dir = output_dir
        self.country = country
        self.test_type = test_type
        self.n_past_years = 6 if country != "mexico" else 4
        self.n_train_years = 15 if country != "mexico" else 10

        # Grid search parameters - same ranges for fair comparison
        self.n_estimators_values = [50, 100, 200]
        self.max_depth_values = [3, 6, 9]

        if model == "xgboost":
            self.learning_rate_values = [0.05, 0.1, 0.3]
        elif model == "randomforest":
            self.learning_rate_values = [None]  # RF doesn't use learning rate
        else:
            raise ValueError(f"Unknown model: {model}")

        # Setup logging
        setup_logging(rank=0)
        self.logger = logging.getLogger(__name__)

        # Setup output
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = self._get_output_filename()
        self.detailed_output_file = self._get_detailed_output_filename()

        # Load existing results for resume functionality
        self.existing_results = self._load_existing_results()

        self.logger.info(
            f"Initialized BaselineGridSearch for {model} on {crop_type} in {country}"
        )
        self.logger.info(f"Results will be saved to: {self.output_file}")
        self.logger.info(
            f"Detailed results will be saved to: {self.detailed_output_file}"
        )

    def _get_output_filename(self) -> str:
        """Generate output filename based on model"""
        filename = f"baseline_grid_search_{self.model}_{self.crop_type}_{self.country}_{self.test_type}.tsv"
        return os.path.join(self.output_dir, filename)

    def _get_detailed_output_filename(self) -> str:
        """Generate detailed output filename for individual R² values"""
        filename = f"baseline_grid_search_{self.model}_{self.crop_type}_{self.country}_{self.test_type}_detailed.json"
        return os.path.join(self.output_dir, filename)

    def _load_existing_results(self) -> pd.DataFrame:
        """Load existing results if file exists"""
        if os.path.exists(self.output_file):
            try:
                df = pd.read_csv(self.output_file, sep="\t")
                self.logger.info(f"Loaded {len(df)} existing results")
                return df
            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _experiment_exists(
        self,
        n_estimators: int,
        max_depth: Optional[int],
        learning_rate: Optional[float],
    ) -> bool:
        """Check if specific experiment already exists"""
        if self.existing_results.empty or "model" not in self.existing_results.columns:
            return False

        # Handle None values for max_depth
        if max_depth is None:
            mask = (
                (self.existing_results["model"] == self.model)
                & (self.existing_results["n_estimators"] == n_estimators)
                & (self.existing_results["max_depth"].isna())
            )
        else:
            mask = (
                (self.existing_results["model"] == self.model)
                & (self.existing_results["n_estimators"] == n_estimators)
                & (self.existing_results["max_depth"] == max_depth)
            )

        # Add learning_rate check for XGBoost
        if learning_rate is not None:
            mask = mask & (self.existing_results["learning_rate"] == learning_rate)

        matching_rows = self.existing_results[mask]
        if matching_rows.empty:
            return False

        # Check if RMSE column exists and has valid data
        if "rmse" not in matching_rows.columns:
            return False

        rmse_list = matching_rows["rmse"].tolist()
        if len(rmse_list) == 0:
            return False

        value = rmse_list[0]
        if pd.isna(value) or not isinstance(value, str):
            return False

        # Only consider experiment completed if it contains the "±" symbol
        return "±" in value

    def _train_and_evaluate(
        self,
        n_estimators: int,
        max_depth: Optional[int],
        learning_rate: Optional[float],
        seed: int = 1234,
    ) -> Tuple[List[float], List[float]]:
        """Train and evaluate model on all test folds"""

        # Read dataset
        if self.country == "usa":
            crop_df = read_usa_dataset(DATA_DIR)
        else:
            crop_df = read_non_us_dataset(DATA_DIR, self.country)

        # Get test years
        if self.test_type == "extreme":
            test_years = EXTREME_YEARS.get(self.country, {}).get(self.crop_type)
            if test_years is None:
                raise ValueError(
                    f"No extreme years found for {self.crop_type} in {self.country}."
                )
        elif self.test_type == "overall":
            test_years = TEST_YEARS
        elif self.test_type == "ahead_pred":
            test_years = TEST_YEARS
        else:
            raise ValueError(f"Unknown test_type: {self.test_type}")

        fold_rmses = []
        fold_stds = []

        for test_year in test_years:
            # Clear CROP_YIELD_STATS for this fold
            CROP_YIELD_STATS[self.crop_type]["mean"].clear()
            CROP_YIELD_STATS[self.crop_type]["std"].clear()

            test_gap = 4 if self.test_type == "ahead_pred" else 0

            # Get numpy data
            (X_train, y_train), (X_test, y_test) = get_numpy_train_test_data(
                crop_df,
                self.n_train_years,
                test_year,
                self.n_past_years,
                self.crop_type,
                self.country,
                test_gap=test_gap,
            )

            # Train model
            if self.model == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=seed,
                    n_jobs=-1,
                )
            elif self.model == "randomforest":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    n_jobs=-1,
                )
            else:
                raise ValueError(f"Unknown model: {self.model}")

            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Compute RMSE (on standardized values)
            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

            # Get the std used for this fold
            fold_std = CROP_YIELD_STATS[self.crop_type]["std"][0]

            fold_rmses.append(rmse)
            fold_stds.append(fold_std)

        return fold_rmses, fold_stds

    def _save_results(
        self,
        n_estimators: int,
        max_depth: Optional[int],
        learning_rate: Optional[float],
        fold_rmses: List[float],
        fold_stds: List[float],
        runtime_seconds: float,
    ):
        """Save experiment results to TSV and JSON files"""

        # Convert RMSE to bu/acre
        rmse_bu_acre = [rmse * std for rmse, std in zip(fold_rmses, fold_stds)]
        avg_rmse = np.mean(rmse_bu_acre)
        std_rmse = np.std(rmse_bu_acre)

        # Compute R²
        r_squared_values = [
            1 - (rmse / std) ** 2 for rmse, std in zip(rmse_bu_acre, fold_stds)
        ]
        avg_r2 = np.mean(r_squared_values)
        std_r2 = np.std(r_squared_values)

        # Load current results
        df = self._load_existing_results()

        # Create new row
        new_row = {
            "model": self.model,
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else np.nan,
            "rmse": f"{avg_rmse:.3f} ± {std_rmse:.3f}",
            "r2": f"{avg_r2:.3f} ± {std_r2:.3f}",
            "runtime_seconds": f"{runtime_seconds:.1f}",
        }

        if learning_rate is not None:
            new_row["learning_rate"] = learning_rate

        # Check if row already exists
        if not df.empty:
            if max_depth is None:
                mask = (
                    (df["model"] == self.model)
                    & (df["n_estimators"] == n_estimators)
                    & (df["max_depth"].isna())
                )
            else:
                mask = (
                    (df["model"] == self.model)
                    & (df["n_estimators"] == n_estimators)
                    & (df["max_depth"] == max_depth)
                )

            if learning_rate is not None:
                mask = mask & (df["learning_rate"] == learning_rate)

            if mask.any():
                # Update existing row
                row_idx = df[mask].index[0]
                for key, value in new_row.items():
                    df.loc[row_idx, key] = value
            else:
                # Append new row
                new_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_df], ignore_index=True)
        else:
            # Create new dataframe
            df = pd.DataFrame([new_row])

        # Save to TSV
        df.to_csv(self.output_file, sep="\t", index=False)

        # Save detailed results
        self._save_detailed_results(
            n_estimators,
            max_depth,
            learning_rate,
            fold_rmses,
            fold_stds,
            r_squared_values,
            runtime_seconds,
        )

        self.logger.info(
            f"Saved results: n_estimators={n_estimators}, max_depth={max_depth}, "
            f"lr={learning_rate}, RMSE={avg_rmse:.3f}±{std_rmse:.3f}, R²={avg_r2:.3f}±{std_r2:.3f}"
        )

    def _save_detailed_results(
        self,
        n_estimators: int,
        max_depth: Optional[int],
        learning_rate: Optional[float],
        fold_rmses: List[float],
        fold_stds: List[float],
        r_squared_values: List[float],
        runtime_seconds: float,
    ):
        """Save detailed results with individual fold values to JSON file"""

        # Load existing detailed results
        detailed_results = self._load_detailed_results()

        # Create experiment key
        lr_str = f"_lr_{learning_rate}" if learning_rate is not None else ""
        md_str = f"{max_depth}" if max_depth is not None else "None"
        experiment_key = (
            f"{self.model}_{self.crop_type}_{self.country}_{self.test_type}_"
            f"n_{n_estimators}_depth_{md_str}{lr_str}"
        )

        # Convert to bu/acre
        rmse_bu_acre = [rmse * std for rmse, std in zip(fold_rmses, fold_stds)]

        detailed_results[experiment_key] = {
            "model": self.model,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "mean_rmse": float(np.mean(rmse_bu_acre)),
            "std_rmse": float(np.std(rmse_bu_acre)),
            "mean_r2": float(np.mean(r_squared_values)),
            "std_r2": float(np.std(r_squared_values)),
            "individual_rmse_bu_acre": [float(x) for x in rmse_bu_acre],
            "individual_r2_values": [float(x) for x in r_squared_values],
            "fold_count": len(fold_rmses),
            "runtime_seconds": runtime_seconds,
        }

        # Save to JSON
        with open(self.detailed_output_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

    def _load_detailed_results(self) -> Dict:
        """Load existing detailed results if file exists"""
        if os.path.exists(self.detailed_output_file):
            try:
                with open(self.detailed_output_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load existing detailed results: {e}")
                return {}
        return {}

    def run(self):
        """Run the complete grid search"""

        total_combinations = (
            len(self.n_estimators_values)
            * len(self.max_depth_values)
            * len(self.learning_rate_values)
        )

        self.logger.info(f"Total parameter combinations to test: {total_combinations}")
        self.logger.info(f"n_estimators values: {self.n_estimators_values}")
        self.logger.info(f"max_depth values: {self.max_depth_values}")
        if self.model == "xgboost":
            self.logger.info(f"learning_rate values: {self.learning_rate_values}")

        completed_experiments = 0
        skipped_experiments = 0

        # Run experiments
        for n_estimators in self.n_estimators_values:
            for max_depth in self.max_depth_values:
                for learning_rate in self.learning_rate_values:
                    # Check if experiment already exists
                    if self._experiment_exists(n_estimators, max_depth, learning_rate):
                        self.logger.info(
                            f"Skipping n_estimators={n_estimators}, max_depth={max_depth}, "
                            f"lr={learning_rate} (already completed)"
                        )
                        skipped_experiments += 1
                        continue

                    self.logger.info(
                        f"Running experiment: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}"
                    )

                    start_time = time.time()
                    try:
                        fold_rmses, fold_stds = self._train_and_evaluate(
                            n_estimators, max_depth, learning_rate
                        )
                        end_time = time.time()
                        runtime_seconds = end_time - start_time

                        # Save results
                        self._save_results(
                            n_estimators,
                            max_depth,
                            learning_rate,
                            fold_rmses,
                            fold_stds,
                            runtime_seconds,
                        )

                        completed_experiments += 1

                    except Exception as e:
                        self.logger.error(
                            f"Failed experiment n_estimators={n_estimators}, max_depth={max_depth}, "
                            f"lr={learning_rate}: {str(e)}",
                            exc_info=True,
                        )

        self.logger.info(f"Grid search completed!")
        self.logger.info(
            f"Completed: {completed_experiments}, Skipped: {skipped_experiments}"
        )
        self.logger.info(f"Results saved to: {self.output_file}")
        self.logger.info(f"Detailed results saved to: {self.detailed_output_file}")


def setup_args() -> argparse.Namespace:
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(
        description="Grid search for baseline models (XGBoost, Random Forest)"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "randomforest"],
        help="Model to use for experiments",
    )

    parser.add_argument(
        "--crop-type",
        required=True,
        choices=[
            "soybean",
            "corn",
            "wheat",
            "sunflower",
            "cotton",
            "sugarcane",
            "beans",
            "corn_rainfed",
            "beans_rainfed",
        ],
        help="Crop type to predict",
    )

    parser.add_argument(
        "--output-dir",
        default="data/grid_search",
        help="Directory to save results (default: data/grid_search)",
    )

    parser.add_argument(
        "--country",
        help="country dataset to use: usa, argentina, brazil, or mexico",
        default="usa",
        type=str,
        choices=["usa", "argentina", "brazil", "mexico"],
    )

    parser.add_argument(
        "--test-type",
        help="type of test evaluation: extreme (extreme years), overall (2014-18), or ahead_pred (2014-18 with 5-year gap)",
        default="extreme",
        type=str,
        choices=["extreme", "overall", "ahead_pred"],
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = setup_args()

    grid_search = BaselineGridSearch(
        model=args.model,
        crop_type=args.crop_type,
        output_dir=args.output_dir,
        country=args.country,
        test_type=args.test_type,
    )

    grid_search.run()


if __name__ == "__main__":
    main()
