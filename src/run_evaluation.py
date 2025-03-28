from argparse import ArgumentParser

from textless.data.speech_encoder import SpeechEncoder
import torch
import os

from metrics import unit_edit_distance_from_two_datasets
from dataloader import AudioDataset


parser = ArgumentParser()

parser.add_argument("--dataset_root",
                    type = str,
                    help = "The path to the folder containing the LibriSpeech directory.",
                    default="LibriSpeech")

parser.add_argument("--dataset_ref",
                    type = str,
                    help = "The reference dataset. Default is test-clean",
                    default="test-clean")

parser.add_argument("--dataset_augmented",
                    type = str,
                    help = "The augmented dataset. Default = test-clean-noisy",
                    default="test-clean-noisy")

parser.add_argument("--encoder",
                    type = str,
                    help = "The encoder used to create the latent representations. Default is hubert-base-ls960.",
                    default="hubert-base-ls960")

parser.add_argument("--k",
                    type = int,
                    help = "The number of clusters for the quantizer.",
                    default=100)

args = parser.parse_args()

def get_store_name(dataset_augmented):
    if dataset_augmented=='test-clean-noisy':
        return 'noisy'
    elif dataset_augmented=='test-clean-pitched':
        return 'pitched'
    elif dataset_augmented=='test-clean-reverb':
        return 'reverb'
    elif dataset_augmented=='test-clean-stretched':
        return 'stretched'
    if dataset_augmented=='test-clean-noisy-hard':
        return 'noisy-hard'
    elif dataset_augmented=='test-clean-pitched-hard':
        return 'pitched-hard'
    elif dataset_augmented=='test-clean-stretched-hard':
        return 'stretched-hard'
    elif dataset_augmented=='test-clean-pitch_shift_resample':
        return 'pitched-shift-resample'   

"""if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SpeechEncoder.by_name(dense_model_name=args.encoder, quantizer_model_name="kmeans", vocab_size=args.k, deduplicate=False, need_f0=False).to(device)
    encoder.eval()

    dataset_file = os.path.join(args.dataset_root, args.dataset_ref)
    dataset_augmented_file = os.path.join(args.dataset_root, args.dataset_augmented)
    print(dataset_file, dataset_augmented_file)
    dataset = AudioDataset(dataset_file)
    dataset_augmented = AudioDataset(dataset_augmented_file)
    print(dataset)

    os.makedirs("results", exist_ok=True)
    store_file = os.path.join("results", get_store_name(args.dataset_augmented) + "_" + args.encoder + f"_{args.k}clusters" + "_results.txt")
    unit_edit_distance_from_two_datasets(dataset, dataset_augmented, encoder, store_file=store_file , verbose=True)"""


if __name__ == "__main__":
    for k in [50, 100, 200]:
        for dataset_augmented_name in ["test-clean-reverb"]:
        #for dataset_augmented_name in ["test-clean-reverb", "test-clean-stretched", "test-clean-pitched", "test-clean-noisy"]:
        #for dataset_augmented_name in ["test-clean-stretched-hard", "test-clean-pitched-hard", "test-clean-noisy-hard"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = SpeechEncoder.by_name(dense_model_name=args.encoder, quantizer_model_name="kmeans", vocab_size=k, deduplicate=True, need_f0=False).to(device)
            encoder.eval()

            dataset_file = os.path.join(args.dataset_root, args.dataset_ref)
            dataset_augmented_file = os.path.join(args.dataset_root, dataset_augmented_name)
            dataset = AudioDataset(dataset_file)
            dataset_augmented = AudioDataset(dataset_augmented_file)

            os.makedirs("results", exist_ok=True)
            store_file = os.path.join("results", get_store_name(dataset_augmented_name) + "_" + args.encoder + f"_{k}clusters" + "_results.txt")
            unit_edit_distance_from_two_datasets(dataset, dataset_augmented, encoder, store_file=store_file , verbose=True)