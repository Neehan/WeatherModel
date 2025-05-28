import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import random

random.seed(1234)
np.random.seed(1234)

torch.use_deterministic_algorithms(True)

from torch_lr_finder import LRFinder

from src.yield_prediction.dataloader import read_soybean_dataset, get_train_test_loaders
from src.yield_prediction.model import YieldPredictor
from src.yield_prediction.cnn_transformer import CNNYieldPredictor
from src.yield_prediction.wf_linear import WFLinearPredictor
from src.yield_prediction.bert_model import BERTYieldPredictor
from src.yield_prediction.train import training_loop
from src.yield_prediction.constants import *
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", help="batch size", default=128, type=int)
parser.add_argument(
    "--n_past_years", help="number of past years to look at", default=6, type=int
)
parser.add_argument(
    "--n_epochs", help="number of training epoches", default=40, type=int
)
parser.add_argument(
    "--init_lr", help="initial learning rate for Adam", default=0.0005, type=float
)
parser.add_argument(
    "--lr_decay_factor",
    help="learning rate exponential decay factor",
    default=0.98,
    type=float,
)
parser.add_argument(
    "--n_warmup_epochs", help="number of warmup epoches", default=5, type=float
)
parser.add_argument(
    "--no-pretraining",
    dest="no_pretraining",
    action="store_true",
    help="don't use the pretrained model",
)

parser.add_argument(
    "--pretrained_model_path",
    help="path to pretrained model weights",
    default=None,
    type=str,
)


parser.add_argument(
    "--model_size",
    help="model size small (2M), medium (8M), and large (56M)",
    default="small",
    type=str,
)

parser.add_argument(
    "--model",
    help="weatherformer, cnn, wflinear, bert",
    default="weatherformer",
    type=str,
)

parser.add_argument(
    "--find_optimal_lr",
    dest="find_optimal_lr",
    action="store_true",
    help="find optimal learning rate using LR finder",
)
