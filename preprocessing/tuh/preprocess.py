from itertools import product
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import mne
import numpy as np
import pandas as pd

from tqdm import tqdm
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.datasets import BaseConcatDataset, BaseDataset
import glob
import re
import os
import shutil
import multiprocessing

mne.set_log_level('WARNING')

class TUHAbnormal(BaseConcatDataset):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.
    Parameters
    ----------
    path: str
        parent directory of the dataset
    subject_ids: list(int) | int
        (list of) int of subject(s) to be read
    target_name: str
        can be 'pathological', 'gender', or 'age'
    preload: bool
        if True, preload the data of the Raw objects.
    """
    def __init__(self, path, subject_ids=None, target_name="pathological",preload=False):
        
        all_file_paths = read_all_file_names(path, extension='.edf', key=self._time_key)
        breakpoint()
        
        if subject_ids is None:
            subject_ids = np.arange(len(all_file_paths))
        
        channels = ['EEG FP2-REF','EEG FP1-REF','EEG F4-REF','EEG F3-REF','EEG C4-REF','EEG C3-REF','EEG P4-REF','EEG P3-REF','EEG O2-REF','EEG O1-REF','EEG F8-REF','EEG F7-REF',
                    #'EEG T4-REF',
                    'EEG T3-REF','EEG T6-REF','EEG T5-REF','EEG A2-REF','EEG A1-REF','EEG FZ-REF','EEG CZ-REF','EEG PZ-REF','EEG LOC-REF','EEG ROC-REF','EEG EKG1-REF','EMG-REF','EEG 26-REF','EEG 27-REF','EEG 28-REF','EEG 29-REF','EEG 30-REF','EEG T1-REF','EEG T2-REF','PHOTIC-REF','IBI','BURSTS','SUPPR']
        
        all_base_ds = []
        for subject_id in subject_ids:
            file_path = all_file_paths[subject_id]
            raw = mne.io.read_raw_edf(file_path, preload=preload)
            chs = raw.ch_names
            exclude_ch = []
            for ch in chs:
                if 'CZ-REF' not in ch:
                    exclude_ch.append(ch)
            raw = mne.io.read_raw_edf(file_path, preload=preload,exclude=exclude_ch)
            path_splits = file_path.split("/")
            if "abnormal" in path_splits:
                pathological = True
            else:
                assert "normal" in path_splits
                pathological = False
            if "train" in path_splits:
                session = "train"
            else:
                assert "eval" in path_splits
                session = "eval"
            age, gender = _parse_age_and_gender_from_edf_header(file_path)
            description = pd.Series(
                {'age': age, 'pathological': pathological, 'gender': gender,
                'session': session, 'subject': subject_id}, name=subject_id)
            base_ds = BaseDataset(raw, description, target_name=target_name)
            all_base_ds.append(base_ds)

        super().__init__(all_base_ds)

    @staticmethod
    def _time_key(file_path):
        # the splits are specific to tuh abnormal eeg data set
        splits = file_path.split('/')
        p = r'(\d{4}_\d{2}_\d{2})'
        [date] = re.findall(p, splits[-2])
        date_id = [int(token) for token in date.split('_')]
        recording_id = _natural_key(splits[-1])
        session_id = re.findall(r'(s\d*)_', (splits[-2]))
        return date_id + session_id + recording_id

# TODO: this is very slow. how to improve?
def read_all_file_names(directory, extension, key):
    """Read all files with specified extension from given path and sorts them
    based on a given sorting key.
    Parameters
    ----------
    directory: str
        file path on HDD
    extension: str
        file path extension, i.e. '.edf' or '.txt'
    key: calable
        sorting key for the file paths
    Returns
    -------
    file_paths: list(str)
        a list to all files found in (sub)directories of path
    """
    assert extension.startswith(".")
    file_paths = glob.glob(directory + '**/*' + extension, recursive=True)
    file_paths = sorted(file_paths, key=key)
    assert len(file_paths) > 0, (
        f"something went wrong. Found no {extension} files in {directory}")
    return file_paths


def _natural_key(string):
    pattern = r'(\d+)'
    key = [int(split) if split.isdigit() else None
           for split in re.split(pattern, string)]
    return key


def _parse_age_and_gender_from_edf_header(file_path, return_raw_header=False):
    assert os.path.exists(file_path), f"file not found {file_path}"
    f = open(file_path, 'rb')
    content = f.read(88)
    f.close()
    if return_raw_header:
        return content
    patient_id = content[8:88].decode('ascii')
    [age] = re.findall("Age:(\d+)", patient_id)
    [gender] = re.findall("\s(\w)\s", patient_id)
    return int(age), gender
    
    
def load_example_data(TUH_PATH,preload, window_len_s, n_subjects=None,n_jobs=1):
    """Create windowed dataset from subjects of the TUH Abnormal dataset.

    Parameters
    ----------
    preload: bool
        If True, use eager loading, otherwise use lazy loading.
    n_subjects: int
        Number of subjects to load.

    Returns
    -------
    windows_ds: BaseConcatDataset
        Windowed data.

    .. warning::
        The recordings from the TUH Abnormal corpus do not all share the same
        sampling rate. The following assumes that the files have already been
        resampled to a common sampling rate.
    """
    if n_subjects!=None:
        subject_ids = list(range(n_subjects))
    else:
        subject_ids = None
        
    ds = TUHAbnormal(
        TUH_PATH, subject_ids=subject_ids, target_name='pathological',
        preload=preload)

    fs = ds.datasets[0].raw.info['sfreq']
    window_len_samples = int(fs * window_len_s)
    window_stride_samples = int(fs * 4)
    # window_stride_samples = int(fs * window_len_s)
    windows_ds = create_fixed_length_windows(
        ds, start_offset_samples=0, stop_offset_samples=None,
        window_size_samples=window_len_samples,
        window_stride_samples=window_stride_samples, drop_last_window=True,
        preload=preload, drop_bad_windows=True,n_jobs=n_jobs)

    # Drop bad epochs
    # XXX: This could be parallelized.
    # XXX: Also, this could be implemented in the Dataset object itself.
    for ds in windows_ds.datasets:
        ds.windows.drop_bad()
        assert ds.windows.preload == preload

    return windows_ds

train_path = '/scratch/tuh/edf/train/'
eval_path = '/scratch/tuh/edf/eval/'
train_save_path = '/scratch/allsamples_tuh/train'
test_save_path = '/scratch/allsamples_tuh/test'


if os.path.exists('/scratch/allsamples_tuh'):
    shutil.rmtree('/scratch/allsamples_tuh')

os.mkdir('/scratch/allsamples_tuh')
os.mkdir(train_save_path)
os.mkdir(test_save_path)

train_dataset = load_example_data(train_path,preload=False,window_len_s=30,n_jobs=12)
eval_dataset = load_example_data(eval_path,preload=False,window_len_s=30,n_jobs=12)

def multi_fc(save_path,ds,start_subject,end_subject):
    for j in tqdm(range(start_subject,end_subject)):
        subject_ds = ds.datasets[j]
        end = len(subject_ds)
        dct = {}
        dct['x'] = []
        dct['y'] = []
        for i in range(0,end):
            sub = ds[i]
            dct['x'].append(torch.nn.functional.interpolate(torch.tensor(sub[0]).unsqueeze(0),scale_factor=3000/7500).numpy().squeeze(0))
            dct['y'].append(sub[1])

        dct['x'] = np.array(dct['x'])
        dct['y'] = np.array(dct['y'])
        dct['x'] = np.transpose(dct['x'],(0,2,1))
        temp_path = os.path.join(save_path, "subject_"+str(i) +'.npz')
        np.savez(temp_path, **dct)

multi_fc(train_save_path,train_dataset,0,len(train_dataset.datasets))
multi_fc(test_save_path,eval_dataset,0,len(eval_dataset.datasets))
