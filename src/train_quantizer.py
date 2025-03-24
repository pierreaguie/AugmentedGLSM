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





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load("/Data/AugGLSMDatasets/train-clean-100_hubert-base-ls960_100_pitch_shift_1.0.pt", weights_only=False)
    latent_dim = 256

    quantizer = Quantizer(latent_dim=latent_dim, hidden_dims=[512, 256], n_clusters=100).to(device)

    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)

    trainer = Trainer(max_epochs=10, progress_bar_refresh_rate=20)
    trainer.fit(quantizer, train_loader)
    
    trainer.save_checkpoint("quantizer.ckpt")
    
    print("Quantizer training complete.")
    
    print("Saving quantizer model...")




        


    