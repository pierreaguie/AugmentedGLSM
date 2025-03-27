import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, root_folder, max_file_size=100*1024*1024):
        """
        Dataset personnalisé pour charger des fichiers audio .flac dans un dossier et ses sous-dossiers.
        
        Args:
            root_folder (str): Le chemin du dossier racine contenant les fichiers .flac.
        """
        self.root_folder = root_folder
        self.file_paths = []
        self.max_file_size=max_file_size

        # Parcourir récursivement les sous-dossiers pour récupérer les fichiers .flac
        for subdir, _, files in os.walk(root_folder):
            for filename in files:
                if filename.endswith(".flac"):
                    self.file_paths.append(os.path.join(subdir, filename))

    def __len__(self):
        """Retourne le nombre de fichiers dans le dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Retourne la waveform et la fréquence d'échantillonnage d'un fichier."""
        file_path = self.file_paths[idx]
        #print(file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate

        
