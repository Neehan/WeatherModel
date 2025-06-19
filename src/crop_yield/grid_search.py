import os
import logging
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from datetime import datetime

from src.crop_yield.yield_main import main
from src.utils.utils import setup_logging

# Pretrained model path mapping - update these paths as needed
PRETRAINED_MODEL_PATHS = {
    "weatherformersinusoid": "data/trained_models/pretraining/weatherformersinusoid_small.pth",
    "weatherformermixture": "data/trained_models/pretraining/weatherformermixture_small.pth",
}


def run_single_experiment(args_dict, gpu_id):
    """Run a single experiment on specified GPU"""
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        args_dict["device"] = f"cuda:{gpu_id}"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Setup logging for this process
    setup_logging(rank=gpu_id)
    logger = logging.getLogger(__name__)

    experiment_name = (
        f"{args_dict['model']}_beta{args_dict['beta']}_"
        f"years{args_dict['n_train_years']}_"
        f"pretrained{args_dict['pretrained_model_path'] is not None}"
    )

    logger.info(f"Starting experiment on GPU {gpu_id}: {experiment_name}")

    try:
        avg_rmse, std_rmse = main(args_dict)

        result = {
            "experiment_name": experiment_name,
            "args": args_dict,
            "avg_rmse": avg_rmse,
            "std_rmse": std_rmse,
            "gpu_id": gpu_id,
            "status": "success",
        }

        logger.info(
            f"Completed experiment {experiment_name}: RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}"
        )
        return result

    except Exception as e:
        logger.error(f"Failed experiment {experiment_name} on GPU {gpu_id}: {str(e)}")
        result = {
            "experiment_name": experiment_name,
            "args": args_dict,
            "avg_rmse": None,
            "std_rmse": None,
            "gpu_id": gpu_id,
            "status": "failed",
            "error": str(e),
        }
        return result


def generate_experiment_configs():
    """Generate all experiment configurations"""

    # Grid search parameters
    beta_values = [0.0, 1e-4, 1e-3, 1e-2]
    n_train_years_values = [5, 10, 20, 30]
    model_configs = [
        {"model": "weatherformersinusoid", "n_mixture_components": 1},
        {"model": "weatherformermixture", "n_mixture_components": 7},
    ]
    pretrained_options = [True, False]

    # Base configuration
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
    }

    experiments = []

    for beta in beta_values:
        for n_train_years in n_train_years_values:
            for model_config in model_configs:
                for use_pretrained in pretrained_options:
                    # Create experiment config
                    config = copy.deepcopy(base_config)
                    config["beta"] = beta
                    config["n_train_years"] = n_train_years
                    config["model"] = model_config["model"]
                    config["n_mixture_components"] = model_config[
                        "n_mixture_components"
                    ]

                    # Set pretrained model path
                    if use_pretrained:
                        config["pretrained_model_path"] = PRETRAINED_MODEL_PATHS[
                            model_config["model"]
                        ]
                    else:
                        config["pretrained_model_path"] = None

                    experiments.append(config)

    return experiments


def run_grid_search(num_gpus=4, max_workers=None):
    """Run grid search across multiple GPUs"""

    setup_logging(rank=0)
    logger = logging.getLogger(__name__)

    # Generate all experiment configurations
    experiments = generate_experiment_configs()

    logger.info(f"Generated {len(experiments)} experiments to run")
    logger.info(f"Using {num_gpus} GPUs")

    # Determine max workers (default to number of GPUs)
    if max_workers is None:
        max_workers = num_gpus

    results = []

    # Run experiments in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_experiment = {}

        for i, experiment in enumerate(experiments):
            gpu_id = i % num_gpus  # Assign GPU in round-robin fashion
            future = executor.submit(run_single_experiment, experiment, gpu_id)
            future_to_experiment[future] = (experiment, gpu_id, i)

        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            experiment, gpu_id, exp_idx = future_to_experiment[future]
            try:
                result = future.result()
                results.append(result)

                logger.info(f"Completed {len(results)}/{len(experiments)} experiments")

            except Exception as e:
                logger.error(f"Experiment {exp_idx} failed with exception: {str(e)}")
                # Still add a failed result
                results.append(
                    {
                        "experiment_name": f"experiment_{exp_idx}",
                        "args": experiment,
                        "avg_rmse": None,
                        "std_rmse": None,
                        "gpu_id": gpu_id,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"grid_search_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Grid search completed! Results saved to {results_file}")

    # Print summary
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]

    logger.info(
        f"Summary: {len(successful_results)} successful, {len(failed_results)} failed"
    )

    if successful_results:
        # Find best result
        best_result = min(successful_results, key=lambda x: x["avg_rmse"])
        logger.info(f"Best result: {best_result['experiment_name']}")
        logger.info(
            f"Best RMSE: {best_result['avg_rmse']:.3f} ± {best_result['std_rmse']:.3f}"
        )

        # Print top 5 results
        top_results = sorted(successful_results, key=lambda x: x["avg_rmse"])[:5]
        logger.info("\nTop 5 results:")
        for i, result in enumerate(top_results, 1):
            logger.info(
                f"{i}. {result['experiment_name']}: {result['avg_rmse']:.3f} ± {result['std_rmse']:.3f}"
            )

    return results


if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available! Running on CPU (will be slow)")
        num_gpus = 1
    else:
        num_gpus = min(4, torch.cuda.device_count())
        print(f"Found {torch.cuda.device_count()} GPUs, using {num_gpus}")

    # Run grid search
    results = run_grid_search(num_gpus=num_gpus)
