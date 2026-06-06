import torch
import torch.nn as nn
import torch.nn.functional as F

class PatientContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, patient_ids):
        device = embeddings.device
        B = embeddings.shape[0]

        # Normalize embeddings
        z = F.normalize(embeddings, dim=1) # (B, D)

        # Cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (B, B)

        labels = labels.view(-1).long() # (B,)
        patient_ids = patient_ids.view(-1).long() # (B,)

        # Masks
        same_patient = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1)) # (B, B)
        same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)) # (B, B)

        # Remove self-pairs
        self_mask = torch.eye(B, dtype=torch.bool, device=device) # (B, B)

        # ----------------------------
        # Define masks
        # ----------------------------

        # Positive: same patient + same class
        pos_mask = same_patient & same_class & ~self_mask # (B, B) 


        # Negative:
        # 1. Same patient, different class
        neg_mask_1 = same_patient & (~same_class) # (B, B)

        # 2. Different patient, different class
        neg_mask_2 = (~same_patient) & (~same_class) # (B, B)

        neg_mask = (neg_mask_1 | neg_mask_2) # (B, B)

        # ----------------------------
        # Compute loss
        # ----------------------------

        exp_sim = torch.exp(sim) # (B, B)

        # For each anchor, sum positives and negatives
        pos_sum = (exp_sim * pos_mask).sum(dim=1) # (B,)
        neg_sum = (exp_sim * neg_mask).sum(dim=1) # (B,)

        # Avoid division by zero
        eps = 1e-8
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + eps) + eps)

        return loss.mean() # (1,)