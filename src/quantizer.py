import torch
import torch.nn as nn

from typing import List


class Quantizer(nn.Module):

    def __init__(self, latent_dim : int, hidden_dims : List[int], n_clusters : int):
        super(Quantizer, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_clusters = n_clusters

        self.layers = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LeakyReLU()
        ])

        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Linear(hidden_dims[-1], n_clusters))
        self.layers.append(nn.LogSoftmax(dim=-1))


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
