import torchaudio
import torch
import os
import librosa

import torchaudio.transforms as T
import numpy as np
import pyroomacoustics as pra
import random
from tqdm import tqdm

from dataloader import AudioDataset


#Utils

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def save_audio(waveform, sample_rate, file_path):
    torchaudio.save(file_path, waveform, sample_rate)


#Transforms

def time_stretch(waveform, sample_rate, low_rate=0.8, high_rate=1.2, n_fft=512, hard_augment=False, device = "cpu"):
    if hard_augment:
        rate = random.choice([low_rate, high_rate])
    else:
        rate = torch.FloatTensor(1).uniform_(low_rate, high_rate).item()
    stft_transform = torch.stft(waveform, n_fft=n_fft, hop_length=n_fft // 2, return_complex=True)
    transform = T.TimeStretch(n_freq=n_fft//2+1).to(device)
    stretched_stft = transform(stft_transform, rate)
    waveform_stretched = torch.istft(stretched_stft, n_fft=n_fft, hop_length=n_fft // 2)
    
    return waveform_stretched

def pitch_shift_resample(waveform, sample_rate, low_rate=0.8, high_rate=1.2, hard_augment=False):
    if hard_augment:
        rate = random.choice([low_rate, high_rate])
    else:
        rate = torch.FloatTensor(1).uniform_(low_rate, high_rate).item()

    new_sample_rate = int(sample_rate * rate)
    resampled_waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform_stretched = torchaudio.functional.resample(resampled_waveform, orig_freq=new_sample_rate, new_freq=sample_rate)

    return waveform_stretched

def pitch_shift(waveform, sample_rate, n_steps=4, hard_augment=False, device = "cpu"):
    if hard_augment:
        n_steps = random.choice([-n_steps, n_steps])
    else:
        n_steps = random.choice([i for i in range(-n_steps, n_steps + 1) if i != 0]) #exclude 0 steps pitch shift (no shift)
    transform = T.PitchShift(sample_rate, n_steps, n_fft=512).to(device)
    return transform(waveform)

"""def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise
"""

def add_noise(waveform, snr_db_range=(5, 15), device = "cpu"):
    signal_power = torch.mean(waveform ** 2)
    snr_db = torch.FloatTensor(1).uniform_(*snr_db_range).item()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform, device=device) * torch.sqrt(noise_power)
    
    return waveform + noise

def add_reverb(waveform: torch.Tensor, sample_rate: int = 4) -> torch.Tensor:

    audio_np = waveform.numpy() if waveform.ndim == 1 else waveform.squeeze(0).numpy()

    #sample randomly room, source, microphone position
    room_dim = np.random.uniform(low=[3, 3, 2], high=[10, 10, 4])
    room = pra.ShoeBox(room_dim, fs=sample_rate, max_order=10, absorption=np.random.uniform(0.1, 0.4))
    source_position = np.random.uniform(low=[0.5, 0.5, 0.5], high=room_dim - np.array([0.5, 0.5, 0.5]))

    while True:
        mic_position = np.random.uniform(low=[0.5, 0.5, 0.5], high=room_dim - np.array([0.5, 0.5, 0.5]))
        if np.linalg.norm(np.array(mic_position) - np.array(source_position)) > 1:  # Éviter qu'ils soient trop proches
            break

    room.add_source(source_position, signal=audio_np)
    room.add_microphone_array(pra.MicrophoneArray(np.array(mic_position).reshape(3, 1), room.fs))

    # compute response
    room.compute_rir()
    rir = room.rir[0][0] 

    # apply convolution
    audio_modified = np.convolve(audio_np, rir, mode='full')
    audio_modified /= np.max(np.abs(audio_modified))

    return torch.tensor(audio_modified, dtype=torch.float32).unsqueeze(0) #shape (1, x)

def clip_waveform(waveform, clip_level=0.5):
    """Clips the waveform up to clip_level*max(waveform)"""
    max_val = torch.max(torch.abs(waveform))
    return torch.clamp(waveform, min=-clip_level*max_val, max=clip_level*max_val)


def lowpass_filter(waveform, sample_rate, cutoff_freq=4000):
    return torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_freq)


def save_transformed_dataset(folder_path="LibriSpeech/test-clean", transforms=['stretched', 'pitched', 'reverb', 'noisy'], hard_augment=False, custom_name=None):
    dataset = AudioDataset(folder_path)

    transformations = {
        "stretched": lambda waveform, sample_rate: time_stretch(waveform, sample_rate, hard_augment),
        "pitched": lambda waveform, sample_rate: pitch_shift(waveform, sample_rate, hard_augment),
        "reverb": lambda waveform, sample_rate: add_reverb(waveform, sample_rate),
        "noisy": lambda waveform, sample_rate: add_noise(waveform),
        "clip_sound": lambda waveform, sample_rate : clip_waveform(waveform),
        "pitch_shift_resample": lambda waveform, sample_rate : pitch_shift_resample(waveform, sample_rate, hard_augment)
    }

    transformations = {key: transformations[key] for key in transforms if key in transformations}

    for transfo_name, transfo_func in transformations.items():
        output_dir = f"{folder_path}-{transfo_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, (waveform, sample_rate) in tqdm(enumerate(dataset)):
            file_path = dataset.file_paths[idx]
            parts = file_path.split(os.sep)
            parts[3] = f'test-clean-{transfo_name}{custom_name}' if custom_name else f'test-clean-{transfo_name}'
            output_file_path = os.sep.join(parts)
            output_folder = os.sep.join(parts[:-1])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            transformed_waveform = transfo_func(waveform, sample_rate)
            transformed_waveform=transformed_waveform.detach()
            save_audio(transformed_waveform, sample_rate, output_file_path)
        print(f'Done for transformation : {transfo_name}')
    print('Done for the whole dataset!')



if __name__ == "__main__":
    save_transformed_dataset(transforms=['pitched'], hard_augment=False, custom_name=None, folder_path="/Data/LibriSpeech/test-clean")