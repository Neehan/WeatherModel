import os
import logging
import torch
import argparse
import pandas as pd
import copy
import json
from typing import Dict, Tuple, Optional, List

from src.crop_yield.yield_main import main as yield_main_func
from src.utils.utils import setup_logging, get_model_params

PRETRAINED_MODEL_PATHS = {
    "weatherformer": "data/trained_models/stored_pretraining/weatherformer_2.0m_latest.pth",
    "weatherautoencoder": "data/trained_models/stored_pretraining/weatherautoencoder_1.9m_latest.pth",
    "weatherformersinusoid": "data/trained_models/stored_pretraining/weatherformer_sinusoid_2.0m_latest.pth",
    "weatherformermixture": "data/trained_models/stored_pretraining/weatherformer_mixture_2.0m_latest.pth",
    "weatherautoencodermixture": "data/trained_models/pretraining/weatherautoencoder_2.0m_latest.pth",
    "weatherautoencodersinusoid": "data/trained_models/pretraining/weatherautoencoder_2.0m_latest.pth",
    "cnnrnn": None,
    "linear": None,
}


class GridSearch:
    """Grid search for crop yield prediction models"""

    def __init__(
        self,
        model: str,
        load_pretrained: bool,
        crop_type: str,
        batch_size: int,
        init_lr: float,
        output_dir: str = "data/grid_search",
    ):
        self.model = model
        self.load_pretrained = load_pretrained
        self.crop_type = crop_type
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.method = "pretrained" if load_pretrained else "not_pretrained"

        # Grid search parameters
        self.beta_values = [0.0, 1e-4, 1e-3]
        self.n_train_years_values = [5, 15, 30]

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
            f"Initialized GridSearch for {model} ({'with' if load_pretrained else 'without'} pretraining) on {crop_type}"
        )
        self.logger.info(f"Results will be saved to: {self.output_file}")
        self.logger.info(
            f"Detailed results will be saved to: {self.detailed_output_file}"
        )

    def _get_output_filename(self) -> str:
        """Generate output filename based on model and pretrained setting"""
        filename = f"grid_search_{self.model}_{self.method}_{self.crop_type}.tsv"
        return os.path.join(self.output_dir, filename)

    def _get_detailed_output_filename(self) -> str:
        """Generate detailed output filename for individual R² values"""
        filename = (
            f"grid_search_{self.model}_{self.method}_{self.crop_type}_detailed.json"
        )
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

    def _experiment_exists(self, beta: float, n_train_years: int) -> bool:
        """Check if specific experiment (beta, year) already exists and completed successfully"""
        if self.existing_results.empty or "model" not in self.existing_results.columns:
            return False

        mask = (
            (self.existing_results["model"] == self.model)
            & (self.existing_results["method"] == self.method)
            & (self.existing_results["beta"] == beta)
        )

        matching_rows = self.existing_results[mask]
        if matching_rows.empty:
            return False

        year_col = f"year_{n_train_years}"
        if year_col not in matching_rows.columns:
            return False

        year_values = matching_rows[year_col].values
        if len(year_values) == 0:
            return False

        # Check if the value contains "±" symbol, indicating successful completion
        value = year_values[0]
        if pd.isna(value) or not isinstance(value, str):
            return False

        # Only consider experiment completed if it contains the "±" symbol
        return "±" in value

    def _get_base_config(self) -> Dict:
        """Get base configuration for experiments"""
        # Model-specific configuration
        if "sinusoid" in self.model:
            n_mixture_components = 1
        elif "mixture" in self.model:
            n_mixture_components = 2
        else:
            n_mixture_components = 1  # Default for other models

        base_config = {
            "batch_size": self.batch_size,
            "n_past_years": 6,
            "n_epochs": 40,
            "init_lr": self.init_lr,
            "decay_factor": None,
            "n_warmup_epochs": 10,
            "model_size": "small",
            "use_optimal_lr": False,
            "seed": 1234,
            "model": self.model,
            "crop_type": self.crop_type,
            "n_mixture_components": n_mixture_components,
            "model_size_params": get_model_params("small"),
        }

        # Set pretrained model path
        if self.load_pretrained:
            base_config["pretrained_model_path"] = PRETRAINED_MODEL_PATHS[self.model]
        else:
            base_config["pretrained_model_path"] = None

        return base_config

    def _run_single_experiment(self, config: Dict) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[List[float]],
    ]:
        """Run a single experiment with given configuration"""
        experiment_name = (
            f"{config['model']}_beta_{config['beta']}_years_{config['n_train_years']}_"
            f"pretrained_{config['pretrained_model_path'] is not None}"
        )

        self.logger.info(f"Starting experiment: {experiment_name}")

        try:
            avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values = yield_main_func(
                config
            )
            self.logger.info(
                f"Completed {experiment_name}: RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}, R² = {avg_r2:.3f} ± {std_r2:.3f}"
            )
            return avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values
        except Exception as e:
            self.logger.error(f"Failed experiment {experiment_name}: {str(e)}")
            return None, None, None, None, None

    def _run_beta_experiments(self, beta: float) -> Dict[
        int,
        Tuple[
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[List[float]],
        ],
    ]:
        """Run experiments for a single beta value across all year values"""
        base_config = self._get_base_config()
        base_config["beta"] = beta

        results = {}

        for n_train_years in self.n_train_years_values:
            # Check if this specific (beta, year) experiment already exists
            if self._experiment_exists(beta, n_train_years):
                self.logger.info(
                    f"Skipping beta={beta}, years={n_train_years} (already completed)"
                )
                continue

            config = copy.deepcopy(base_config)
            config["n_train_years"] = n_train_years

            avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values = (
                self._run_single_experiment(config)
            )
            results[n_train_years] = (
                avg_rmse,
                std_rmse,
                avg_r2,
                std_r2,
                r_squared_values,
            )

        return results

    def _save_results(
        self,
        beta: float,
        results: Dict[
            int,
            Tuple[
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[List[float]],
            ],
        ],
    ):
        """Save experiment results to TSV file and detailed JSON file"""
        if not results:  # No new results to save
            return

        # Load current results
        df = self._load_existing_results()

        # Find existing row for this beta
        if not df.empty and "model" in df.columns:
            mask = (
                (df["model"] == self.model)
                & (df["method"] == self.method)
                & (df["beta"] == beta)
            )

            if mask.any():
                # Update existing row with only the new results
                row_idx = df[mask].index[0]
                for n_years, (
                    mean_rmse,
                    std_rmse,
                    mean_r2,
                    std_r2,
                    r_squared_values,
                ) in results.items():
                    year_col = f"year_{n_years}"
                    year_r2_col = f"year_{n_years}_r2"
                    if mean_rmse is not None and std_rmse is not None:
                        df.loc[row_idx, year_col] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
                        df.loc[row_idx, year_r2_col] = f"{mean_r2:.3f} ± {std_r2:.3f}"
                    else:
                        df.loc[row_idx, year_col] = "FAILED"
                        df.loc[row_idx, year_r2_col] = "FAILED"
            else:
                # Create new row with only the attempted experiments
                new_row = {"model": self.model, "method": self.method, "beta": beta}
                for n_years, (
                    mean_rmse,
                    std_rmse,
                    mean_r2,
                    std_r2,
                    r_squared_values,
                ) in results.items():
                    year_col = f"year_{n_years}"
                    year_r2_col = f"year_{n_years}_r2"
                    if mean_rmse is not None and std_rmse is not None:
                        new_row[year_col] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
                        new_row[year_r2_col] = f"{mean_r2:.3f} ± {std_r2:.3f}"
                    else:
                        new_row[year_col] = "FAILED"
                        new_row[year_r2_col] = "FAILED"

                new_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_df], ignore_index=True)
        else:
            # Empty DataFrame - create new row with only attempted experiments
            new_row = {"model": self.model, "method": self.method, "beta": beta}
            for n_years, (
                mean_rmse,
                std_rmse,
                mean_r2,
                std_r2,
                r_squared_values,
            ) in results.items():
                year_col = f"year_{n_years}"
                year_r2_col = f"year_{n_years}_r2"
                if mean_rmse is not None and std_rmse is not None:
                    new_row[year_col] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
                    new_row[year_r2_col] = f"{mean_r2:.3f} ± {std_r2:.3f}"
                else:
                    new_row[year_col] = "FAILED"
                    new_row[year_r2_col] = "FAILED"

            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

        # Save to file
        df.to_csv(self.output_file, sep="\t", index=False)

        # Save detailed results with individual R² values
        self._save_detailed_results(beta, results)

        self.logger.info(f"Saved results for beta={beta}")

    def _save_detailed_results(
        self,
        beta: float,
        results: Dict[
            int,
            Tuple[
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[List[float]],
            ],
        ],
    ):
        """Save detailed results with individual R² values to JSON file"""
        # Load existing detailed results
        detailed_results = self._load_detailed_results()

        # Create experiment key
        experiment_key = f"{self.model}_{self.method}_{self.crop_type}_beta_{beta}"

        if experiment_key not in detailed_results:
            detailed_results[experiment_key] = {}

        # Save individual R² values for each year configuration
        for n_years, (
            mean_rmse,
            std_rmse,
            mean_r2,
            std_r2,
            r_squared_values,
        ) in results.items():
            year_key = f"year_{n_years}"
            if r_squared_values is not None:
                detailed_results[experiment_key][year_key] = {
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                    "mean_r2": mean_r2,
                    "std_r2": std_r2,
                    "individual_r2_values": r_squared_values,
                    "fold_count": len(r_squared_values),
                }
            else:
                detailed_results[experiment_key][year_key] = {
                    "mean_rmse": None,
                    "std_rmse": None,
                    "mean_r2": None,
                    "std_r2": None,
                    "individual_r2_values": None,
                    "fold_count": 0,
                    "status": "FAILED",
                }

        # Save detailed results to JSON file
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
        # Set device info
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            self.logger.info(f"Using GPU: {device}")
        else:
            self.logger.info("CUDA not available! Running on CPU (will be slow)")

        self.logger.info(f"Generated {len(self.beta_values)} beta values to test")
        self.logger.info(f"Model: {self.model}, Method: {self.method}")

        completed_experiments = 0
        skipped_experiments = 0

        # Run experiments for each year, then each beta within that year
        for n_train_years in self.n_train_years_values:
            self.logger.info(
                f"\n=== Running experiments for {n_train_years} training years ==="
            )

            for beta in self.beta_values:
                # Skip beta > 0 for weatherautoencoder, cnnrnn, and linear since they don't use beta parameter
                if (
                    self.model in ["weatherautoencoder", "cnnrnn", "linear"]
                    and beta > 0
                ):
                    self.logger.info(
                        f"Skipping beta={beta} for {self.model} (doesn't use beta parameter)"
                    )
                    skipped_experiments += 1
                    continue

                # Check if this specific (beta, year) experiment already exists
                if self._experiment_exists(beta, n_train_years):
                    self.logger.info(
                        f"Skipping beta={beta}, years={n_train_years} (already completed)"
                    )
                    skipped_experiments += 1
                    continue

                self.logger.info(
                    f"Running experiment: beta={beta}, years={n_train_years}"
                )

                # Run single experiment
                base_config = self._get_base_config()
                base_config["beta"] = beta
                base_config["n_train_years"] = n_train_years

                avg_rmse, std_rmse, avg_r2, std_r2, r_squared_values = (
                    self._run_single_experiment(base_config)
                )

                # Save result immediately
                results = {
                    n_train_years: (
                        avg_rmse,
                        std_rmse,
                        avg_r2,
                        std_r2,
                        r_squared_values,
                    )
                }
                self._save_results(beta, results)

                completed_experiments += 1
                self.logger.info(
                    f"Completed and saved: beta={beta}, years={n_train_years}"
                )

        self.logger.info(f"Grid search completed!")
        self.logger.info(
            f"Completed: {completed_experiments}, Skipped: {skipped_experiments}"
        )
        self.logger.info(f"Results saved to: {self.output_file}")
        self.logger.info(
            f"Detailed results with individual R² values saved to: {self.detailed_output_file}"
        )


def setup_args() -> argparse.Namespace:
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(
        description="Grid search for crop yield prediction models"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "weatherformer",
            "weatherformersinusoid",
            "weatherformermixture",
            "weatherautoencodermixture",
            "weatherautoencodersinusoid",
            "weatherautoencoder",
            "cnnrnn",
            "linear",
        ],
        help="Model to use for experiments",
    )

    parser.add_argument(
        "--load-pretrained", action="store_true", help="Use pretrained model weights"
    )

    parser.add_argument(
        "--crop-type",
        required=True,
        choices=["soybean", "corn", "winter_wheat"],
        help="Crop type to predict: soybean, corn or winter_wheat",
    )

    parser.add_argument(
        "--output-dir",
        default="data/grid_search",
        help="Directory to save results (default: data/grid_search)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )

    parser.add_argument(
        "--init-lr",
        type=float,
        default=0.0005,
        help="Initial learning rate (default: 0.0005)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = setup_args()

    grid_search = GridSearch(
        model=args.model,
        load_pretrained=args.load_pretrained,
        crop_type=args.crop_type,
        batch_size=args.batch_size,
        init_lr=args.init_lr,
        output_dir=args.output_dir,
    )

    grid_search.run()


if __name__ == "__main__":
    main()
