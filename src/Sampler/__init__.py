import torch
from torch.utils.data import Sampler
import random

class PatientSampler(Sampler):
    def __init__(self, dataset, patient_ids, number_of_patients_per_batch=2,
                 number_of_classes=2, batch_size=32, seed=None):
        
        total_slots = number_of_patients_per_batch * number_of_classes
        if batch_size % total_slots != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_patients x num_classes = {total_slots}"
            )

        self.dataset = dataset
        self.number_of_patients_per_batch = number_of_patients_per_batch
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.per_patient_per_class = batch_size // total_slots
        self.rng = random.Random(seed)

        self.patient_map = {}
        for patient in patient_ids:
            sr = torch.where((dataset.labels == 0) & (dataset.patient_ids == patient))[0].tolist()
            af = torch.where((dataset.labels == 1) & (dataset.patient_ids == patient))[0].tolist()
            if len(sr) >= self.per_patient_per_class and len(af) >= self.per_patient_per_class:
                self.patient_map[patient] = {"sr": sr, "af": af}

        # Upper-bound estimate;
        total_servings = sum(
            min(len(v["sr"]), len(v["af"])) // self.per_patient_per_class
            for v in self.patient_map.values()
        )
        self._approx_total_batches = total_servings // number_of_patients_per_batch

    def __len__(self):
        return self._approx_total_batches

    def __iter__(self):
        local_map = {}
        for p, indices in self.patient_map.items():
            sr = list(indices["sr"]); self.rng.shuffle(sr)
            af = list(indices["af"]); self.rng.shuffle(af)
            local_map[p] = {"sr": sr, "af": af}

        active = set(local_map.keys())
        n = self.per_patient_per_class

        while len(active) >= self.number_of_patients_per_batch:
            selected = self.rng.sample(sorted(active), self.number_of_patients_per_batch)
            batch = []
            for patient in selected:
                batch.extend(local_map[patient]["sr"][-n:])
                del local_map[patient]["sr"][-n:]
                batch.extend(local_map[patient]["af"][-n:])
                del local_map[patient]["af"][-n:]
                if len(local_map[patient]["sr"]) < n or len(local_map[patient]["af"]) < n:
                    active.discard(patient)
            yield batch