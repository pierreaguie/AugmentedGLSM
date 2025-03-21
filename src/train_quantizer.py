from textless.data.speech_encoder import SpeechEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

from quantizer import Quantizer



def train_quantizer_epoch(quantizer : Quantizer, dataloader : DataLoader, optimizer : Optimizer):
    """
    Train the quantizer for one epoch.

    Args:
        quantizer (Quantizer): The quantizer to train.
        dataloader (DataLoader): The dataloader for the dataset.
        optimizer (Optimizer): The optimizer to use.

    Returns:
        loss_epoch: The loss of the epoch.
    """

    quantizer.train()

    loss_epoch = 0.
    
    for i, (x_aug, units) in enumerate(dataloader):

        units_aug = quantizer(x_aug)
        loss_step = F.ctc_loss(units, units_aug)
        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()
        loss_epoch += loss_step.item()

    return loss_epoch




        


    