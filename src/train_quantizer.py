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

from dataset import augment_dataset, collate_fn

from argparse import ArgumentParser

from functools import partial

from torchaudio.datasets import LIBRISPEECH
from textless.data.speech_encoder import SpeechEncoder
from transform import pitch_shift, time_stretch, add_reverb, add_noise, clip_waveform, lowpass_filter

parser = ArgumentParser()

parser.add_argument("--dataset_root",
                    type = str,
                    help = "The path to the folder containing the LibriSpeech directory.",
                    default="/Data")

parser.add_argument("--split",
                    type = str,
                    help = "The split to use for the dataset. Either train-clean-100 or test-clean.",
                    default="train-clean-100")

parser.add_argument("--encoder",
                    type = str,
                    help = "The encoder used to create the latent representations. Default is hubert-base-ls960.",
                    default="hubert-base-ls960")

parser.add_argument("--k",
                    type = int,
                    help = "The number of clusters for the quantizer.",
                    default=100)

parser.add_argument("--augmentation",
                    type = str,
                    help = "The augmentation function to use. Possible options: pitch_shift, time_stretch, reverb, noise, clip, lowpass.",
                    default="pitch_shift")

parser.add_argument("--augment_parameter",
                    type = str,
                    help = "The parameter for the augmentation function. Either 'default' or 'UED30' ",
                    default="default")

parser.add_argument("--save_root",
                    type = str,
                    help = "The root directory where the augmented dataset is saved. If None, the dataset is not saved.",
                    default="/Data/AugGLSMDatasets")

parser.add_argument("--batch_size",
                    type = int,
                    help = "The batch size for training the quantizer.",
                    default=64)

parser.add_argument("--max_epochs",
                    type = int,
                    help = "The maximum number of epochs for training the quantizer.",
                    default=50)

parser.add_argument("--lr",
                    type = float,
                    help = "The learning rate for training the quantizer.",
                    default=1e-3)


def get_augmentation(augmentation, augment_parameter):
    fs = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if augmentation == "pitch_shift":
        if augment_parameter == "default":
            return lambda x : pitch_shift(x, fs, n_steps = 8, hard_augment=True, device = device)
        elif augment_parameter == "UED30":
            n_steps = 5
            return lambda x : pitch_shift(x, fs, n_steps, hard_augment=True, device = device)
        
    elif augmentation == "time_stretch":
        if augment_parameter == "default":
            return lambda x : time_stretch(x, fs, hard_augment=True, device = device)
        elif augment_parameter == "UED30":
            low_rate = .92
            high_rate = 1.08
            return lambda x : time_stretch(x, fs, low_rate, high_rate, hard_augment=True, device = device)
        
    elif augmentation == "reverb":
        return lambda x : add_reverb(x.cpu(), fs)
    
    elif augmentation == "noise":
        if augment_parameter == "default":
            return lambda x : add_noise(x, device = device)
        elif augment_parameter == "UED30":
            snr_db_range = (15, 18)
            return lambda x : add_noise(x, snr_db_range, device = device)
    
    elif augmentation == "clip":
        thresh = .065
        return lambda x : clip_waveform(x, clip_level=thresh)
        
    elif augmentation == "lowpass":
        cutoff_freq = 4000
        return lambda x : lowpass_filter(x, sample_rate=fs, cutoff_freq=cutoff_freq)
    
    else:
        return lambda x : x
    



if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SpeechEncoder.by_name(dense_model_name=args.encoder, quantizer_model_name="kmeans", vocab_size=args.k, deduplicate=True, need_f0=False).to(device)
    encoder.eval()

    librispeech = LIBRISPEECH(root=args.dataset_root, url=args.split)
    augmentation = get_augmentation(args.augmentation, args.augment_parameter)
    dataset_save_path = f"{args.save_root}/{args.split}_{args.encoder}_{args.k}_{args.augmentation}_{args.augment_parameter}.pt"
    dataset = augment_dataset(librispeech, encoder, augmentation, dataset_save_path)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

    latent_dim = 768

    quantizer = Quantizer(latent_dim=latent_dim, hidden_dims=[512, 256], n_clusters=args.k, lr = args.lr).to(device)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    log_path = f"lightning_logs/{args.split}_{args.encoder}_{args.k}_{args.augmentation}_{args.augment_parameter}"
    logger = TensorBoardLogger("lightning_logs", name=f"{args.split}_{args.encoder}_{args.k}_{args.augmentation}_{args.augment_parameter}")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=log_path, filename='quantizer-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min')

    trainer = Trainer(max_epochs=args.max_epochs, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(quantizer, train_loader, val_loader)

    print("Training complete.")
    print(f"Quantizer saved at {log_path}.")


        


    