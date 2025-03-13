import torch
from torch.utils.data import Dataset, TensorDataset

from textless.data.collater_utils import collate_tensors
from textless.data.speech_encoder import SpeechEncoder

from tqdm import tqdm

from typing import Optional, Callable


class AugmentedDataset(Dataset):
    def __init__(self, units, dense):
        self.units = units
        self.dense = dense

    def __len__(self):
        return len(self.units)

    def __getitem__(self, idx):
        return self.units[idx], self.dense[idx]


@torch.no_grad()
def augment_dataset(dataset : Dataset, encoder : SpeechEncoder, augmentation : Callable[[torch.Tensor], torch.Tensor],
                    path : Optional[str] = None) -> Dataset:
    """
    Create an augmented dataset, with discrete representations of the original audio, and dense representations of the augmented audio, unsing a speech encoder and audio transformation function.

    Args:
        dataset (Dataset): The original dataset.
        encoder (SpeechEncoder): The speech encoder.
        augmentation (Callable): The audio transformation function.
        path (Optional[str]): The path to save the augmented dataset. Defaults to None. 

    Returns:
        augmented_dataset: The augmented dataset.
    """

    encoder.eval()
    n = len(dataset)
    units = []
    dense = []

    for i in tqdm(range(n), desc = "Creating augmented dataset"):
        x = dataset[i][0].to(encoder.device)
        x_aug = augmentation(x)

        z = encoder(x)
        z_units = z['units']
        units.append(z_units)

        z_aug = encoder(x_aug)
        z_aug_dense = z_aug['dense']
        dense.append(z_aug_dense)
        
    augmented_dataset = AugmentedDataset(units, dense)

    if path:
        torch.save(augmented_dataset, path)

    return augmented_dataset
