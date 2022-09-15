import torch
import numpy as np
import os
from torch.utils.data import Dataset
from .augmentations import augment


class Load_Pretext_Dataset(Dataset):

    # Initialize your data, download, etc.
    def __init__(self, data_path, config):
        super(Load_Pretext_Dataset, self).__init__()
        self.data_path = data_path
        self.files = os.listdir(self.data_path)
        self.files = [
            os.path.join(self.data_path, file) for file in self.files
        ]
        self.config = config

    def __getitem__(self, index):
        x = torch.tensor(np.load(self.files[index])["pos"])
        x = x[0][0].unsqueeze(0)  # 1,3000
        return augment(x, self.config)

    def __len__(self):
        return len(self.files)


def data_generator(data_path, configs):

    train_dataset = Load_Pretext_Dataset(data_path, configs)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=10,
    )
    return train_loader


class TuneDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects):
        self.subjects = subjects
        self._add_subjects()

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self):
        return self.X.shape[0]

    def _add_subjects(self):
        self.X = []
        self.y = []

        for subject in self.subjects:
            self.X.append(np.expand_dims(subject["x"][:, :, 0], axis=1))
            self.y.append(subject["y"])

        self.X = torch.tensor(np.concatenate(self.X, axis=0))
        self.y = torch.tensor(np.concatenate(self.y, axis=0)).long()
