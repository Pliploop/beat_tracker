import soundfile as sf
import torchaudio
import numpy as np
import torch

def load_audio(path, target_seconds = None, start = None, target_sr = None):
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
                audio = torchaudio.functional.resample(audio, sr, target_sr)
            
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                
            return audio,sr
        except Exception as e:
            print(e)
            return None
            
        
        
def get_spectrogram(audio, target_sr = 22050, n_fft = 2048, n_mels = 128, hop_length = 512):
    # get the spectrogram of the audio
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=n_fft, n_mels=n_mels,hop_length = hop_length)(audio)
    return spec