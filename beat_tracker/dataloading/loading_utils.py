import soundfile as sf
import torchaudio
import numpy as np

def load_audio(path, target_seconds = None, start = None, target_sr = None):
        try:
            info = sf.info(path)
            sr = info.samplerate
            if target_seconds is not None:
                
                new_target_samples = int(target_seconds * sr)
                # load a random segment of the audio
                if start is None:
                    start = np.random.randint(0, info.frames - new_target_samples)
                else:
                    start = int(start * sr)
                
                audio, _ = torchaudio.load(path, frame_offset=start, num_frames=new_target_samples)
            else:
                audio, _ = torchaudio.load(path)
                
            if target_sr is not None:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
                
            return audio,sr
        except Exception as e:
            print(e)
            return None
            
        
        
def get_spectrogram(self, audio, target_sr = 22050, n_fft = 2048, n_mels = 128):
    # get the spectrogram of the audio
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=n_fft, n_mels=n_mels)(audio)
    return spec