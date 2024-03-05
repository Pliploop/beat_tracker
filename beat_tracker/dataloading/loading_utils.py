import soundfile as sf
import torchaudio
import numpy as np
import torch

def load_audio(path, target_seconds=None, start=None, target_sr=None, mono=True):
    """
    Load audio from a file.

    Args:
        path (str): The path to the audio file.
        target_seconds (float, optional): The target duration of the audio in seconds. If specified, a random segment of the audio will be loaded with the specified duration. Defaults to None.
        start (float, optional): The start position in seconds from where to load the audio. If specified, the audio will be loaded starting from this position. Defaults to None.
        target_sr (int, optional): The target sample rate of the audio. If specified, the audio will be resampled to the target sample rate. Defaults to None.
        mono (bool, optional): Whether to convert the audio to mono. If True, the audio will be converted to mono. If False, the audio will be kept as is. Defaults to True.

    Returns:
        tuple: A tuple containing the loaded audio tensor and the sample rate of the audio.
            - audio (torch.Tensor): The loaded audio tensor.
            - sr (int): The sample rate of the audio.

    Raises:
        Exception: If there is an error loading the audio.

    """
    try:
        info = sf.info(path)
        extension = path.split(".")[-1]
        sr = info.samplerate
        if extension == "mp3":
            n_frames = info.frames - 8192
        else:
            n_frames = info.frames
        if target_seconds is not None:
            new_target_samples = int(target_seconds * sr)
            # load a random segment of the audio
            if start is None:
                start = np.random.randint(0, n_frames - new_target_samples)
            else:
                start = int(start * sr)
            audio, _ = sf.read(path, start=start, stop=start + new_target_samples)
        else:
            audio, _ = sf.read(path)
        audio = torch.tensor(audio, dtype=torch.float32)
        if target_sr is not None:
            # transpose audio for torchaudio
            original_len_audio_s = audio.shape[0] / sr
            audio = audio.t()
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            audio = audio.t()
            new_len_audio_s = audio.shape[0] / target_sr
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio, sr
    except Exception as e:
        print(e)
        return None
            
        
        
def get_spectrogram(audio, target_sr = 22050, n_fft = 2048, n_mels = 128, hop_length = 512):
    """
    Compute the spectrogram of the given audio.

    Parameters:
    audio (torch.Tensor): The input audio waveform.
    target_sr (int): The target sample rate of the audio (default: 22050).
    n_fft (int): The number of FFT points (default: 2048).
    n_mels (int): The number of Mel filterbanks (default: 128).
    hop_length (int): The hop length for the STFT (default: 512).

    Returns:
    torch.Tensor: The spectrogram of the audio.
    """
    
    # get the spectrogram of the audio
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=n_fft, n_mels=n_mels,hop_length = hop_length)(audio)
    return spec