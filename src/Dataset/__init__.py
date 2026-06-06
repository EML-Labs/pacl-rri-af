import torch
from torch.utils.data import Dataset
import pickle
import hashlib
import json
import os
import h5py
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, 
        processed_dataset_path: str, 
        minimum_af_length: int,
        minimum_sr_length: int,
        window_size:int,
        stride:int,
        buffer_before_af:int=int(60*60),
        length_of_sr_window:int=int(60*60),
        length_of_af_window:int=int(60*60),
        minimum_sr_time_to_be_considered_for_scaler:int=int(60*60),
        train: bool = False,
        test: bool = False,
    ):
        """
        stride: stride between window starts in seconds.
        window_size: number of samples per window after resampling (e.g. 60).
        """

        dataset_properties = {
            "dataset_type": "classification",
            "minimum_af_length": minimum_af_length,
            "minimum_sr_length": minimum_sr_length,
            "window_size": window_size,
            "stride": stride,
            "buffer_before_af": buffer_before_af,
            "length_of_sr_window": length_of_sr_window,
            "length_of_af_window": length_of_af_window,
            "minimum_sr_time_to_be_considered_for_scaler": minimum_sr_time_to_be_considered_for_scaler
        }
        dataset_string = json.dumps(dataset_properties, sort_keys=True)
        file_name_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]
        print(f"Dataset path: {os.path.join(processed_dataset_path, f'{file_name_hash}_train.h5')}")
        if train:
            self.dataset_path = os.path.join(processed_dataset_path, f"{file_name_hash}_train.h5")
        elif test:
            self.dataset_path = os.path.join(processed_dataset_path, f"{file_name_hash}_test.h5")
        else:
            self.dataset_path = os.path.join(processed_dataset_path, f"{file_name_hash}_validation.h5")

        self.segments = []
        self.labels = []
        self.patient_ids = []

        with h5py.File(self.dataset_path, 'r') as f:
            self.patient_ids = f['patient_ids'][:]
            self.segments = f['segments'][:]
            self.labels = f['labels'][:]

        self.segments = torch.from_numpy(self.segments)
        self.labels = torch.from_numpy(self.labels)
        self.patient_ids = torch.from_numpy(self.patient_ids)

    def get_patient_ids(self):
        return np.unique(self.patient_ids)
    
    def get_patient_data(self, patient_id):
        mask = self.patient_ids == patient_id
        return self.segments[mask], self.labels[mask], self.patient_ids[mask]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx], self.patient_ids[idx]
