import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.text import EditDistance

from typing import List


class Quantizer(pl.LightningModule):

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

        self.UED = EditDistance()


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x
    

    def training_step(self, batch, batch_idx):
        units, x_aug, lengths = batch
        units_aug = self(x_aug)
        print(units.shape, units_aug.shape, x_aug.shape)
        loss = F.ctc_loss(units * mask, units_aug * mask)
        result = pl.TrainResult(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return result
    
    def validation_step(self, batch, batch_idx):
        units, x_aug, lengths = batch
        bs = units.shape[0]
        units_aug = self(x_aug)

        print(units.shape, units_aug.shape, x_aug.shape)

        loss = F.ctc_loss(units_aug.view(-1, bs, self.n_clusters), units)
        result = pl.EvalResult(checkpoint_on=loss)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        ued = self.UED(units * mask, units_aug * mask)
        self.log('val_ued', ued, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer