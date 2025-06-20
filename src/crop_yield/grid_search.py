import os
import logging
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from datetime import datetime

from src.crop_yield.yield_main import main
from src.utils.utils import setup_logging, get_model_params

# Pretrained model path mapping - update these paths as needed
PRETRAINED_MODEL_PATHS = {
    "weatherformersinusoid": "data/trained_models/pretraining/weatherformer_sinusoid_2.0m_latest.pth",
    "weatherformermixture": "data/trained_models/pretraining/weatherformer_mixture_2.1m_latest.pth",
}

CHECKPOINT_DIR = "data/grid_search"
CHECKPOINT_FILE = "grid_search_checkpoint.json"


def ensure_checkpoint_dir():
    """Ensure checkpoint directory exists"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(results, completed_indices, total_experiments):
    """Save current progress to checkpoint file"""
    ensure_checkpoint_dir()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

    # Create simplified results format for checkpoint
    simplified_results = []
    for result in results:
        simplified_result = (
            result.copy()
        )  # All results are now in simplified format already
        simplified_results.append(simplified_result)

    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "completed_indices": list(completed_indices),
        "total_experiments": total_experiments,
        "results": simplified_results,
        "progress": f"{len(completed_indices)}/{total_experiments}",
    }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Checkpoint saved: {len(completed_indices)}/{total_experiments} experiments completed"
    )


def load_checkpoint():
    """Load checkpoint if it exists"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        logger = logging.getLogger(__name__)
        logger.info(f"Resuming from checkpoint: {checkpoint_data['progress']}")

        return (
            set(checkpoint_data["completed_indices"]),
            checkpoint_data["results"],
            checkpoint_data["total_experiments"],
        )

    return set(), [], 0


def run_single_experiment(args_dict, gpu_id, experiment_idx):
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

    logger.info(
        f"Starting experiment {experiment_idx} on GPU {gpu_id}: {experiment_name}"
    )

    avg_rmse, std_rmse = main(args_dict)

    # Create simplified result format
    result = args_dict.copy()  # Start with all parameters
    result.update(
        {
            "experiment_idx": experiment_idx,
            "mean_rmse": avg_rmse,
            "std_rmse": std_rmse,
            "gpu_id": gpu_id,
            "status": "success",
        }
    )

    logger.info(
        f"Completed experiment {experiment_idx} ({experiment_name}): RMSE = {avg_rmse:.3f} ± {std_rmse:.3f}"
    )
    return result


def generate_experiment_configs():
    """Generate all experiment configurations with indices"""

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

    # Add model size parameters
    model_size_params = get_model_params("small")
    base_config["model_size_params"] = model_size_params

    experiments = []
    experiment_idx = 0

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

                    experiments.append((experiment_idx, config))
                    experiment_idx += 1

    return experiments


def run_grid_search(num_gpus=4, max_workers=None, checkpoint_frequency=5):
    """Run grid search across multiple GPUs with checkpointing"""

    setup_logging(rank=0)
    logger = logging.getLogger(__name__)

    # Generate all experiment configurations
    all_experiments = generate_experiment_configs()
    total_experiments = len(all_experiments)

    # Load checkpoint if exists
    completed_indices, results, checkpoint_total = load_checkpoint()

    if checkpoint_total > 0 and checkpoint_total != total_experiments:
        logger.warning(
            f"Checkpoint has {checkpoint_total} experiments but current config has {total_experiments}. Starting fresh."
        )
        completed_indices, results = set(), []

    # Filter out completed experiments - this handles non-sequential completion correctly
    remaining_experiments = [
        (idx, config) for idx, config in all_experiments if idx not in completed_indices
    ]

    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Already completed: {len(completed_indices)}")
    logger.info(f"Remaining: {len(remaining_experiments)}")
    logger.info(f"Using {num_gpus} GPUs")

    if not remaining_experiments:
        logger.info("All experiments already completed!")
        return results

    # Determine max workers (default to number of GPUs)
    if max_workers is None:
        max_workers = num_gpus

    # Run remaining experiments in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit remaining experiments
        future_to_experiment = {}

        for i, (experiment_idx, experiment_config) in enumerate(remaining_experiments):
            gpu_id = i % num_gpus  # Assign GPU in round-robin fashion
            future = executor.submit(
                run_single_experiment, experiment_config, gpu_id, experiment_idx
            )
            future_to_experiment[future] = (experiment_config, gpu_id, experiment_idx)

        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            experiment_config, gpu_id, experiment_idx = future_to_experiment[future]

            result = future.result()  # This will raise exception if experiment failed
            results.append(result)
            completed_indices.add(experiment_idx)

            logger.info(
                f"Completed experiment {experiment_idx}: {len(completed_indices)}/{total_experiments} total"
            )

            # Save checkpoint every N completions
            if len(completed_indices) % checkpoint_frequency == 0:
                save_checkpoint(results, completed_indices, total_experiments)

    # Final checkpoint save
    save_checkpoint(results, completed_indices, total_experiments)

    # Save final results to timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = os.path.join(
        CHECKPOINT_DIR, f"grid_search_results_{timestamp}.json"
    )

    with open(final_results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Grid search completed! Final results saved to {final_results_file}")

    # Print summary - all results are successful now
    logger.info(f"Summary: {len(results)} experiments completed successfully")

    if results:
        # Find best result
        best_result = min(results, key=lambda x: x["mean_rmse"])
        logger.info(
            f"Best result: {best_result['model']}_beta{best_result['beta']}_years{best_result['n_train_years']}_pretrained{best_result['pretrained_model_path'] is not None}"
        )
        logger.info(
            f"Best RMSE: {best_result['mean_rmse']:.3f} ± {best_result['std_rmse']:.3f}"
        )

        # Print top 5 results
        top_results = sorted(results, key=lambda x: x["mean_rmse"])[:5]
        logger.info("\nTop 5 results:")
        for i, result in enumerate(top_results, 1):
            experiment_name = f"{result['model']}_beta{result['beta']}_years{result['n_train_years']}_pretrained{result['pretrained_model_path'] is not None}"
            logger.info(
                f"{i}. {experiment_name}: {result['mean_rmse']:.3f} ± {result['std_rmse']:.3f}"
            )

    # Clean up checkpoint file after successful completion
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint file cleaned up after successful completion")

    return results


if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available! Running on CPU (will be slow)")
        num_gpus = 1
    else:
        num_gpus = min(4, torch.cuda.device_count())
        print(f"Found {torch.cuda.device_count()} GPUs, using {num_gpus}")

    # Run grid search with checkpointing
    results = run_grid_search(num_gpus=num_gpus, checkpoint_frequency=3)
