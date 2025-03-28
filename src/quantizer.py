import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.text import EditDistance

from typing import List


class Quantizer(pl.LightningModule):

    def __init__(self, latent_dim : int, hidden_dims : List[int], n_clusters : int, lr : float = 1e-3):
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
        self.layers.append(nn.Linear(hidden_dims[-1], n_clusters + 1))
        
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.lr = lr


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.logsoftmax(x)
        return x
    

    def training_step(self, batch, batch_idx):
        units, x_aug, lengths_units, lengths_dense = batch
        bs = x_aug.shape[0]
        units_aug = self(x_aug)
        log_probs = units_aug.view(-1, bs, self.n_clusters + 1)
        loss = F.ctc_loss(log_probs=log_probs, targets=units, input_lengths=lengths_dense, target_lengths=lengths_units)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        return loss
    

    def validation_step(self, batch, batch_idx):
        units, x_aug, lengths_units, lengths_dense = batch
        bs = x_aug.shape[0]
        units_aug = self(x_aug)
        log_probs = units_aug.view(-1, bs, self.n_clusters + 1)
        loss = F.ctc_loss(log_probs=log_probs, targets=units, input_lengths=lengths_dense, target_lengths=lengths_units)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
