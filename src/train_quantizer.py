import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

from quantizer import Quantizer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import AugmentedDataset, collate_fn





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load("/Data/AugGLSMDatasets/train-clean-100_hubert-base-ls960_100_pitch_shift_1.0.pt", weights_only=False)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

    latent_dim = 768

    quantizer = Quantizer(latent_dim=latent_dim, hidden_dims=[512, 256], n_clusters=100).to(device)

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    trainer = Trainer(max_epochs=10)
    trainer.fit(quantizer, train_loader, val_loader)
    
    trainer.save_checkpoint("quantizer.ckpt")
    
    print("Quantizer training complete.")
    
    print("Saving quantizer model...")




        


    