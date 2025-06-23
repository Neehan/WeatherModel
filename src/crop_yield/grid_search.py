import os
import logging
import torch
import argparse
import pandas as pd
from datetime import datetime
import copy
from typing import Dict, List, Tuple, Optional

from src.crop_yield.yield_main import main as yield_main_func
from src.utils.utils import setup_logging, get_model_params

# Pretrained model path mapping - update these paths as needed
PRETRAINED_MODEL_PATHS = {
    "weatherformersinusoid": "data/trained_models/pretraining/weatherformer_sinusoid_2.0m_latest.pth",
    "weatherformermixture": "data/trained_models/pretraining/weatherformer_mixture_2.1m_latest.pth",
    "weatherautoencodermixture": "data/trained_models/pretraining/weatherautoencoder_1.9m_latest.pth",
    "weatherautoencodersinusoid": "data/trained_models/pretraining/weatherautoencoder_1.9m_latest.pth",
    "weatherautoencoder": "data/trained_models/pretraining/weatherautoencoder_1.9m_latest.pth",
}


class GridSearch:
    """Grid search for crop yield prediction models"""

    def __init__(
        self, model: str, load_pretrained: bool, output_dir: str = "data/grid_search"
    ):
        self.model = model
        self.load_pretrained = load_pretrained
        self.output_dir = output_dir
        self.method = "pretrained" if load_pretrained else "not_pretrained"

        # Grid search parameters
        self.beta_values = [0.0, 1e-4, 1e-3, 1e-2]
        self.n_train_years_values = [5, 10, 20, 30]

        # Setup logging
        setup_logging(rank=0)
        self.logger = logging.getLogger(__name__)

        # Setup output
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = self._get_output_filename()

        # Load existing results for resume functionality
        self.existing_results = self._load_existing_results()

        self.logger.info(
            f"Initialized GridSearch for {model} ({'with' if load_pretrained else 'without'} pretraining)"
        )
        self.logger.info(f"Results will be saved to: {self.output_file}")

    def _get_output_filename(self) -> str:
        """Generate output filename based on model and pretrained setting"""
        filename = f"grid_search_{self.model}_{self.method}.tsv"
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
            n_mixture_components = 7
        else:
            n_mixture_components = 1  # Default for other models

        base_config = {
            "batch_size": 64,
            "n_past_years": 6,
            "n_epochs": 40,
            "init_lr": 0.0005,
            "decay_factor": 0.95,
            "n_warmup_epochs": 10,
            "model_size": "small",
            "use_optimal_lr": False,
            "seed": 1234,
            "model": self.model,
            "n_mixture_components": n_mixture_components,
            "model_size_params": get_model_params("small"),
        }

        # Set pretrained model path
        if self.load_pretrained:
            base_config["pretrained_model_path"] = PRETRAINED_MODEL_PATHS[self.model]
        else:
            base_config["pretrained_model_path"] = None

        return base_config

    def _run_single_experiment(
        self, config: Dict
    ) -> Tuple[Optional[float], Optional[float]]:
        """Run a single experiment with given configuration"""
        experiment_name = (
            f"{config['model']}_beta_{config['beta']}_years_{config['n_train_years']}_"
            f"pretrained_{config['pretrained_model_path'] is not None}"
        )

        self.logger.info(f"Starting experiment: {experiment_name}")

        try:
            avg_rmse, std_rmse = yield_main_func(config)
            self.logger.info(
                f"Completed {experiment_name}: RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}"
            )
            return avg_rmse, std_rmse
        except Exception as e:
            self.logger.error(f"Failed experiment {experiment_name}: {str(e)}")
            return None, None

    def _run_beta_experiments(
        self, beta: float
    ) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
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

            avg_rmse, std_rmse = self._run_single_experiment(config)
            results[n_train_years] = (avg_rmse, std_rmse)

        return results

    def _save_results(
        self, beta: float, results: Dict[int, Tuple[Optional[float], Optional[float]]]
    ):
        """Save experiment results to TSV file"""
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
                for n_years, (mean_rmse, std_rmse) in results.items():
                    year_col = f"year_{n_years}"
                    if mean_rmse is not None and std_rmse is not None:
                        df.loc[row_idx, year_col] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
                    else:
                        df.loc[row_idx, year_col] = "FAILED"
            else:
                # Create new row with only the attempted experiments
                new_row = {"model": self.model, "method": self.method, "beta": beta}
                for n_years, (mean_rmse, std_rmse) in results.items():
                    year_col = f"year_{n_years}"
                    if mean_rmse is not None and std_rmse is not None:
                        new_row[year_col] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
                    else:
                        new_row[year_col] = "FAILED"

                new_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_df], ignore_index=True)
        else:
            # Empty DataFrame - create new row with only attempted experiments
            new_row = {"model": self.model, "method": self.method, "beta": beta}
            for n_years, (mean_rmse, std_rmse) in results.items():
                year_col = f"year_{n_years}"
                if mean_rmse is not None and std_rmse is not None:
                    new_row[year_col] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
                else:
                    new_row[year_col] = "FAILED"

            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

        # Save to file
        df.to_csv(self.output_file, sep="\t", index=False)
        self.logger.info(f"Saved results for beta={beta}")

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

                avg_rmse, std_rmse = self._run_single_experiment(base_config)

                # Save result immediately
                results = {n_train_years: (avg_rmse, std_rmse)}
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


def setup_args() -> argparse.Namespace:
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(
        description="Grid search for crop yield prediction models"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "weatherformersinusoid",
            "weatherformermixture",
            "weatherautoencodermixture",
            "weatherautoencodersinusoid",
            "weatherautoencoder",
        ],
        help="Model to use for experiments",
    )

    parser.add_argument(
        "--load-pretrained", action="store_true", help="Use pretrained model weights"
    )

    parser.add_argument(
        "--output-dir",
        default="data/grid_search",
        help="Directory to save results (default: data/grid_search)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = setup_args()

    grid_search = GridSearch(
        model=args.model,
        load_pretrained=args.load_pretrained,
        output_dir=args.output_dir,
    )

    grid_search.run()


if __name__ == "__main__":
    main()
