import torchaudio
import torch

def reverb(waveform : torch.Tensor, 
           rir_path : str, 
           wet : float) -> torch.Tensor:
    """
    Add reverb to an audio signal.
    """
    rir, _ = torchaudio.load(rir_path)
    rir = rir[0]
    rir = rir / torch.norm(rir, p=2)
    rir = rir.to(waveform.device)
    waveform = torch.nn.functional.conv1d(waveform.unsqueeze(0), rir.unsqueeze(0), padding=0).squeeze(0)
    waveform = waveform * (1 - wet) + wet * torch.nn.functional.pad(waveform, (0, len(rir)))
    return waveform