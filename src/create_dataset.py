from argparse import ArgumentParser

from textless.data.speech_encoder import SpeechEncoder
import torch
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample, PitchShift, AddNoise, TimeStretch
import torchaudio.functional as F

from dataset import augment_dataset
from transform import add_reverb, add_noise

from functools import partial

import os


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
                    help = "The augmentation function to use. Possible options: pitch_shift, time_stretch, reverb, noise.",
                    default="pitch_shift")

parser.add_argument("--augment_parameter",
                    type = float,
                    help = "The parameter for the augmentation function. For pitch_shift, number of semitones.\
                          For time_stretch, stretch factor. For reverb, wet level from 0 to 1. For noise, SNR.",
                    default=1)

parser.add_argument("--save_root",
                    type = str,
                    help = "The root directory where the augmented dataset is saved. If None, the dataset is not saved.",
                    default="/Data/AugGLSMDatasets")

args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SpeechEncoder.by_name(dense_model_name=args.encoder, quantizer_model_name="kmeans", vocab_size=args.k, deduplicate=True, need_f0=False).to(device)
    encoder.eval()
    dataset = LIBRISPEECH(root=args.dataset_root, url=args.split)

    if args.augmentation == "pitch_shift":
        n_steps = int(args.augment_parameter)
        augmentation = PitchShift(16000, n_steps).to(device)
    else:
        raise ValueError("Invalid augmentation")

    if args.save_root:
        if not(os.path.exists(f"{args.save_root}")):
            os.makedirs(f"{args.save_root}")
        save_path = f"{args.save_root}/{args.split}_{args.encoder}_{args.k}_{args.augmentation}_{args.augment_parameter}.pt"
    else:
        save_path = None
        
    augmented_dataset = augment_dataset(dataset = dataset, encoder = encoder, augmentation = augmentation, path = save_path)
    print(f"Augmented dataset saved at {save_path}.")