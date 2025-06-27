import logging
import os
import random
from typing import Any, Dict, List, Type

import numpy as np
import torch

from src.base_models.base_model import BaseModel
from src.base_trainer.base_trainer import BaseTrainer


class CrossValidator:
    """
    K-fold cross validation implementation using BaseTrainer infrastructure.

    This class handles k-fold cross validation by creating trainer instances
    for each fold and leveraging the trainer's get_dataloaders method with
    cross_validation_k parameter.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_kwargs: Dict[str, Any],
        trainer_class: Type[BaseTrainer],
        trainer_kwargs: Dict[str, Any],
        k_folds: int = 5,
    ):
        """
        Initialize CrossValidator.

        Args:
            trainer_class: BaseTrainer subclass to use for training
            trainer_kwargs: Arguments to pass to trainer constructor
            k_folds: Number of folds for cross validation
            random_seed: Random seed for reproducible fold generation
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.trainer_class = trainer_class
        self.trainer_kwargs = trainer_kwargs
        self.k_folds = k_folds
        self.logger = logging.getLogger(__name__)

    def run_cross_validation(self, use_optimal_lr: bool = True) -> Dict[str, Any]:
        """
        Run k-fold cross validation.

        Args:
            use_optimal_lr: Whether to find optimal learning rate for each fold

        Returns:
            Dictionary containing aggregated results across all folds
        """
        self.logger.info(f"Starting {self.k_folds}-fold cross validation")

        fold_results = []
        fold_training_stats = []
        total_best_val_loss = 0.0

        for fold in range(self.k_folds):
            self.logger.info(f"Starting fold {fold + 1}/{self.k_folds}")

            # Reset all random seeds to ensure identical behavior across folds
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)
            torch.cuda.manual_seed(1234)
            torch.use_deterministic_algorithms(True)

            # Create model for this fold
            model = self.model_class(**self.model_kwargs)
            if fold == 0:
                self.logger.info(str(model))

            # Create trainer for this fold
            trainer = self.trainer_class(model=model, **self.trainer_kwargs)
            best_loss = trainer.train(use_optimal_lr=use_optimal_lr)

            # Extract results from this fold
            fold_results.append(best_loss)

            # Extract training statistics if available (for yield prediction)
            training_stats = getattr(trainer, "training_stats", None)
            if training_stats:
                fold_training_stats.append(training_stats)

            total_best_val_loss += best_loss

            self.logger.info(
                f"Fold [{fold + 1} / {self.k_folds}] completed. Best val loss: {best_loss:.4f}"
            )

        # Aggregate results across all folds
        aggregated_results = self._aggregate_results(
            fold_results, total_best_val_loss, fold_training_stats
        )

        self.logger.info(
            f"Cross validation completed. Average best val loss: {aggregated_results['avg_best_val_loss']:.4f}"
        )

        return aggregated_results

    def _aggregate_results(
        self,
        fold_results: List[float],
        total_best_val_loss: float,
        fold_training_stats: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate results across all folds.

        Args:
            fold_results: List of best validation losses from each fold
            total_best_val_loss: Sum of best validation losses across folds
            fold_training_stats: List of training statistics from each fold

        Returns:
            Aggregated results dictionary
        """
        n_folds = len(fold_results)

        # Calculate averages
        avg_best_val_loss = total_best_val_loss / n_folds

        # Calculate standard deviation of best validation losses
        std_best_val_loss = np.std(fold_results)

        # Calculate average training statistics if available
        avg_training_stats = {}
        if fold_training_stats:
            for stat in fold_training_stats[0]:
                avg_training_stats[stat] = np.mean(
                    [stats[stat] for stats in fold_training_stats]
                )

        return {
            "avg_best_val_loss": avg_best_val_loss,
            "std_best_val_loss": std_best_val_loss,
            "fold_results": fold_results,
            "n_folds": n_folds,
            "avg_training_stats": avg_training_stats,
        }
