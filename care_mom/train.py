import wandb
import numpy as np
import torch
import argparse
import os
from utils.dataloader import pretext_data
from helper_train import sleep_pretrain
from torch.utils.data import DataLoader
from config import Config
from utils.utils import *

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", type=str, default="care_mom", help="Name for the saved weights"
)
parser.add_argument(
    "--data_dir", type=str, default="/scratch/sleepkfold_allsamples", help="Path to the data"
)
parser.add_argument(
    "--save_path", type=str, default="./saved_weights", help="Path to save weights"
)

args = parser.parse_args()

name = args.name
ss_wandb = wandb.init(
    project="carev2",
    name=name,
    notes="curr to curr loss; surr to surr loss; curr to surr loss",
    save_code=True,
    entity="sleep-staging",
)
config = Config(ss_wandb)

config.src_path = args.data_dir
config.exp_path = args.save_path

config.le_path = "/scratch/new_shhs/test"

ss_wandb.save("./config.py")
ss_wandb.save("./trainer.py")
ss_wandb.save("./data_preprocessing/*")
ss_wandb.save("./models/*")


PRETEXT_FILE = os.listdir(os.path.join(config.src_path, "pretext"))
PRETEXT_FILE.sort(key=natural_keys)
PRETEXT_FILE = [os.path.join(config.src_path, "pretext", f) for f in PRETEXT_FILE]

TEST_FILE = os.listdir(os.path.join(config.le_path))
TEST_FILE.sort(key=natural_keys)
TEST_FILE = [os.path.join(config.le_path, f) for f in TEST_FILE]

print(f'Number of pretext files: {len(PRETEXT_FILE)}')
print(f'Number of test records: {len(TEST_FILE)}')

pretext_loader = DataLoader(pretext_data(config,PRETEXT_FILE), batch_size=config.batch_size, shuffle=True, num_workers=10)

test_records = [np.load(f) for f in TEST_FILE]
test_subjects = dict()

for i, rec in enumerate(test_records):
    if rec['_description'][0] not in test_subjects.keys():
        test_subjects[rec['_description'][0]] = [rec]
    else:
        test_subjects[rec['_description'][0]].append(rec)

test_subjects = list(test_subjects.values())

model = sleep_pretrain(config, name, pretext_loader,test_subjects, ss_wandb)
ss_wandb.watch([model], log="all", log_freq=500)

model.fit()

ss_wandb.finish()
