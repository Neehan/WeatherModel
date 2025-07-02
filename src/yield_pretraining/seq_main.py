#!/usr/bin/env python3
"""
Training script for the sequence-based yield prediction model.
"""

import argparse
from typing import Optional
import torch
import logging
from src.yield_pretraining.models.seq_model import SeqModel
from src.yield_pretraining.trainers.seq_trainer import SeqTrainer
from src.yield_pretraining.dataloader.seq_dataloader import SeqDataloader
from src.utils.utils import setup_logging, parse_args

# Setup argument parser
parser = argparse.ArgumentParser(
    description="Train sequence-based yield prediction model"
)
parser.add_argument(
    "--crop-type",
    default="corn",
    type=str,
    help="Type of crop (e.g., corn, winter_wheat)",
)
parser.add_argument(
    "--batch-size", default=32, type=int, help="Batch size for training"
)
parser.add_argument(
    "--test-year-cutoff",
    default=2009,
    type=int,
    help="Years > this value go to test set",
)
parser.add_argument(
    "--n-past-years",
    default=5,
    type=int,
    help="Number of past years to use for prediction",
)
parser.add_argument(
    "--num-epochs", default=50, type=int, help="Number of training epochs"
)
parser.add_argument("--init-lr", default=5e-4, type=float, help="Initial learning rate")
parser.add_argument(
    "--num-warmup-epochs", default=5, type=int, help="Number of warmup epochs"
)
parser.add_argument(
    "--decay-factor", default=0.95, type=float, help="Learning rate decay factor"
)


def main():
    # Setup logging
    setup_logging(rank=0)  # Single GPU, rank always 0
    logger = logging.getLogger(__name__)

    # Parse arguments
    args_dict = parse_args(parser)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = SeqModel(name=f"seq_{args_dict['crop_type']}_yield_model", device=device)
    logger.info(f"Created model with {model.total_params_formatted()} parameters")

    # Create dataloader
    dataloader = SeqDataloader(
        crop_type=args_dict["crop_type"],
        batch_size=args_dict["batch_size"],
        test_year_cutoff=args_dict["test_year_cutoff"],
        n_past_years=args_dict["n_past_years"],
    )

    # Create trainer
    trainer = SeqTrainer(
        model=model,
        dataloader=dataloader,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["num_epochs"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["num_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
    )

    # Train the model
    logger.info("Starting training...")
    best_val_loss = trainer.train()
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        raise
