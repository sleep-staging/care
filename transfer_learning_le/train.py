import wandb
import numpy as np
import torch
import argparse
import os
from utils.dataloader import pretext_data
from helper_train import run
from torch.utils.data import DataLoader
from config import Config
from utils.utils import *

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--name",
                    type=str,
                    default="me",
                    help="Name for the saved weights")
parser.add_argument(
    "--data_dir",
    type=str,
    default="/scratch/new_shhs",
    help="Path to the data",
)
parser.add_argument("--save_path",
                    type=str,
                    default="./saved_weights",
                    help="Path to save weights")

args = parser.parse_args()

name = args.name

config = Config()

config.src_path = args.data_dir
config.exp_path = os.path.join(args.save_path, name)

if not os.path.exists(config.exp_path):
    os.makedirs(config.exp_path, exist_ok=True)

config.le_path = "/scratch/sleepkfold_allsamples/test"


TEST_FILE = os.listdir(os.path.join(config.le_path))
TEST_FILE.sort(key=natural_keys)
TEST_FILE = [os.path.join(config.le_path, f) for f in TEST_FILE]

print(f"Number of test records: {len(TEST_FILE)}")

test_records = [np.load(f) for f in TEST_FILE]
test_subjects = dict()

for i, rec in enumerate(test_records):
    if rec["_description"][0] not in test_subjects.keys():
        test_subjects[rec["_description"][0]] = [rec]
    else:
        test_subjects[rec["_description"][0]].append(rec)

test_subjects = list(test_subjects.values())

run(config, name, test_subjects)
