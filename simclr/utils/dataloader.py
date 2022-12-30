"""Augmentations pipeline for self-supervised pre-training. 

Two variations of augmentations are used. Jittering, where random uniform noise is added to the EEG signal depending on its peak-to-peak values, 
along with masking, where signals are masked randomly. Flipping, where the EEG signal is horizontally flipped randomly, and scaling, where EEG 
signal is scaled with Gaussian noise.

This file can also be imported as a module and contains the following:

    * Load_Dataset - Loads the dataset and applies the augmentations.
    * data_generator - Generates a dataloader for the dataset.
    * cross_data_generator - Generates a k-fold dataloader for the given dataset. 
"""
__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"

import numpy as np
import copy
import torch

from torch.utils.data import Dataset
from utils.augmentations import augment


class pretext_data(Dataset):

    def __init__(self, config, filepath):
        super(pretext_data, self).__init__()

        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))
        self.config = config

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):

        path = self.file_path[index]
        data = np.load(path)
        pos = torch.tensor(data["pos"][:, :1, :])  # (7, 1, 3000)
        anc = copy.deepcopy(pos)

        # augment
        for i in range(pos.shape[0]):
            pos[i], anc[i] = augment(pos[i], self.config)
        return pos[:, 0, :], anc[:, 0, :]  # (7, 3000)


class TuneDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects):
        self.subjects = subjects
        self._add_subjects()

    def __getitem__(self, index):

        X = self.X[index, :1, :]
        y = self.y[index]
        return X, y

    def __len__(self):
        return self.X.shape[0]

    def _add_subjects(self):
        self.X = []
        self.y = []
        for subject in self.subjects:
            self.X.append(subject["windows"])
            self.y.append(subject["y"])
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
