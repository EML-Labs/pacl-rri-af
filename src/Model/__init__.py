import torch
from torch import nn
import torch.nn.functional as F
from .Encoder import Encoder

class Model(nn.Module):
    def __init__(self, latent_dim:int,projection_dim:int, dropout:float=0.1):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
            )

    def forward(self, rr_window:torch.Tensor)->torch.Tensor:
        """
        rr_window: [B, window_size]
        output: [B, projection_dim], [B, LATENT_SIZE]
        """
        latents = self.encoder(rr_window)
        embeddings = self.projection(latents)
        embeddings = F.normalize(embeddings, dim=1)
        latents = F.normalize(latents, dim=1)
        return embeddings, latents # [B, projection_dim], [B, LATENT_SIZE]
