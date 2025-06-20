import os
import logging
import torch
import copy
import json
from datetime import datetime

from src.crop_yield.yield_main import main
from src.utils.utils import setup_logging, get_model_params

# Pretrained model path mapping
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

    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "completed_indices": list(completed_indices),
        "total_experiments": total_experiments,
        "results": results,
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


def generate_experiment_configs():
    """Generate all experiment configurations"""
    beta_values = [0.0, 1e-4, 1e-3, 1e-2]
    n_train_years_values = [5, 10, 20, 30]
    model_configs = [
        {"model": "weatherformersinusoid", "n_mixture_components": 1},
        {"model": "weatherformermixture", "n_mixture_components": 7},
    ]
    pretrained_options = [True, False]

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

    model_size_params = get_model_params("small")
    base_config["model_size_params"] = model_size_params

    experiments = []
    experiment_idx = 0

    for beta in beta_values:
        for n_train_years in n_train_years_values:
            for model_config in model_configs:
                for use_pretrained in pretrained_options:
                    config = copy.deepcopy(base_config)
                    config["beta"] = beta
                    config["n_train_years"] = n_train_years
                    config["model"] = model_config["model"]
                    config["n_mixture_components"] = model_config[
                        "n_mixture_components"
                    ]

                    if use_pretrained:
                        config["pretrained_model_path"] = PRETRAINED_MODEL_PATHS[
                            model_config["model"]
                        ]
                    else:
                        config["pretrained_model_path"] = None

                    experiments.append((experiment_idx, config))
                    experiment_idx += 1

    return experiments


def run_grid_search(num_gpus=4, checkpoint_frequency=5):
    """Run grid search - simple sequential execution with GPU cycling"""
    setup_logging(rank=0)
    logger = logging.getLogger(__name__)

    # Generate all experiments
    all_experiments = generate_experiment_configs()
    total_experiments = len(all_experiments)

    # Load checkpoint
    completed_indices, results, checkpoint_total = load_checkpoint()

    if checkpoint_total > 0 and checkpoint_total != total_experiments:
        logger.warning(f"Config changed. Starting fresh.")
        completed_indices, results = set(), []

    # Filter remaining experiments
    remaining_experiments = [
        (idx, config) for idx, config in all_experiments if idx not in completed_indices
    ]

    logger.info(
        f"Total: {total_experiments}, Completed: {len(completed_indices)}, Remaining: {len(remaining_experiments)}"
    )

    if not remaining_experiments:
        logger.info("All experiments completed!")
        return results

    # Simple loop - run experiments one by one
    for i, (experiment_idx, config) in enumerate(remaining_experiments):
        gpu_id = i % num_gpus

        # Set device
        if torch.cuda.is_available():
            config["device"] = f"cuda:{gpu_id}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            config["device"] = "cpu"

        experiment_name = f"{config['model']}_beta{config['beta']}_years{config['n_train_years']}_pretrained{config['pretrained_model_path'] is not None}"

        logger.info(
            f"Starting experiment {experiment_idx} on GPU {gpu_id}: {experiment_name}"
        )

        # Run the experiment
        avg_rmse, std_rmse = main(config)

        # Store result
        result = config.copy()
        result.update(
            {
                "experiment_idx": experiment_idx,
                "mean_rmse": avg_rmse,
                "std_rmse": std_rmse,
                "gpu_id": gpu_id,
            }
        )

        results.append(result)
        completed_indices.add(experiment_idx)

        logger.info(
            f"Completed {experiment_idx}: RMSE = {avg_rmse:.3f} ± {std_rmse:.3f} ({len(completed_indices)}/{total_experiments})"
        )

        # Save checkpoint
        if len(completed_indices) % checkpoint_frequency == 0:
            save_checkpoint(results, completed_indices, total_experiments)

    # Final save
    save_checkpoint(results, completed_indices, total_experiments)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = os.path.join(CHECKPOINT_DIR, f"grid_search_results_{timestamp}.json")

    with open(final_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Grid search completed! Results saved to {final_file}")

    if results:
        best_result = min(results, key=lambda x: x["mean_rmse"])
        logger.info(
            f"Best: {best_result['model']}_beta{best_result['beta']}_years{best_result['n_train_years']}"
        )
        logger.info(
            f"Best RMSE: {best_result['mean_rmse']:.3f} ± {best_result['std_rmse']:.3f}"
        )

    # Clean up checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available! Running on CPU")
        num_gpus = 1
    else:
        num_gpus = min(4, torch.cuda.device_count())
        print(f"Using {num_gpus} GPUs")

    results = run_grid_search(num_gpus=num_gpus, checkpoint_frequency=3)
