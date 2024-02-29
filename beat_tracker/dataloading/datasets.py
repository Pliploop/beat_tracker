# create a dataset for the beat tracker, using the ballroom dataset

import os
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset
from beat_tracker.dataloading.loading_utils import load_audio, get_spectrogram


class BeatTrackingDataset(Dataset):
    """_summary_
    
    A dataset for the beat tracker, using the ballroom dataset.
    The audio is loaded and the spectrogram is computed on the fly.

    Args:
        annotations (pd.DataFrame): A dataframe with the columns 'file_path', 'beats', 'downbeats'
        target_sr (int): The target sample rate
        target_seconds (int): The length of the spectrogram in seconds
        n_fft (int): The number of fft points
        fps (int): The frames per second of the spectrogram
        n_mels (int): The number of mel bands
        train (bool): Whether the dataset is for training or not
        augmentations (None): Augmentations to apply to the audio
        transform (None): whether to apply a transform to the audio
    """
    
    
    def __init__(self,
                 annotations,
                 target_sr=22050,
                 target_seconds = 6,
                 n_fft = 2048,
                 fps = 100,
                 n_mels = 128,
                 train = True,
                 augmentations = None,
                 transform=None):
        self.annotations = annotations
        self.transform = transform
        self.target_sr = target_sr
        self.target_seconds = target_seconds
        self.hop_length = target_sr // fps
        self.fps = fps
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.train = train
        self.augmentations = augmentations
        
        
    def __len__(self):
        return len(self.annotations)
    
    def dummy_call(self):
        dummy_audio = torch.randn(1, self.target_sr * self.target_seconds)
        dummy_beats = torch.zeros(self.target_sr * self.target_seconds)
        dummy_downbeats = torch.zeros(self.target_sr * self.target_seconds)
        dummy_spectrogram = torch.randn( self.n_mels, self.target_sr * self.target_seconds // self.hop_length)
        return {
            'audio': dummy_audio,
            'spectrogram': dummy_spectrogram,
            'beats': dummy_beats,
            'downbeats': dummy_downbeats,
            'original_sample_rate': self.target_sr,
            'new_sample_rate': self.target_sr,
            'fps': self.fps,
        }
        
    
    def __getitem__(self, index):
        
        
        x = self.annotations.iloc[index]
        path = x['file_path']
        beats = x['beats']
        downbeats = x['downbeats']

        # load the audio file
        audio,sr = load_audio(path, target_sr = self.target_sr)
        if audio is None:
            return self[index+1]
        
        spectrogram = get_spectrogram(audio = audio, target_sr = self.target_sr, hop_length = self.hop_length, n_fft = self.n_fft, n_mels = self.n_mels)
        # self.fps frames per second spectrogram.
        
        #beats and downbeats is a list of times in seconds. Convert to binary sequence the size of the spectrogram
        # 1 if there is a beat, 0 otherwise
        beat_sequence = np.zeros(spectrogram.shape[-1])
        downbeat_sequence = np.zeros(spectrogram.shape[-1])
        for beat in beats:
            try:
                beat_sequence[int(beat * self.fps)] = 1
            except:
                print(beat)
        for downbeat in downbeats:
            try:
                downbeat_sequence[int(downbeat * self.fps)] = 1
            except:
                print(downbeat)
            
        # apply a smoothing filter to the beat sequence with convolution and a gaussian kernel
        beat_sequence = np.convolve(beat_sequence, np.array([0.25, 0.5, 1, 0.5, 0.25]), mode='same')
        downbeat_sequence = np.convolve(downbeat_sequence, np.array([0.25, 0.5, 1, 0.5, 0.25]), mode='same')
        
        
        # convert to tensor
        beat_sequence = torch.tensor(beat_sequence, dtype=torch.float32)
        downbeat_sequence = torch.tensor(downbeat_sequence, dtype=torch.float32)
        
        #truncate audio, spectrogram,beat and downbeat sequence to a random chunk of self.target_seconds
        if self.target_seconds is not None:
            start = np.random.randint(0, spectrogram.shape[-1] - self.target_seconds * self.fps)
            audio = audio[:, start:start + self.target_seconds * self.target_sr]
            spectrogram = spectrogram[:, :, start:start + self.target_seconds * self.fps]
            beat_sequence = beat_sequence[start:start + self.target_seconds * self.fps]
            downbeat_sequence = downbeat_sequence[start:start + self.target_seconds * self.fps]
        
        # return the audio and the label
        return {
            'audio': audio,
            'spectrogram': spectrogram,
            'beats': beat_sequence,
            'downbeats': downbeat_sequence,
            'original_sample_rate': sr,
            'new_sample_rate': self.target_sr,
            'fps': self.fps,
        }



class BeatTrackingMelDataset(Dataset):
    """_summary_
    
    A dataset for the beat tracker, using the ballroom dataset.
    Precomputed mel spectrograms are used as input.

    Args:
        annotations (pd.DataFrame): A dataframe with the columns 'file_path', 'beats', 'downbeats', 'sr', 'fps'
        target_sr (int): The target sample rate
        target_seconds (int): The length of the spectrogram in seconds
        fps (int): The frames per second of the spectrogram
    """
    def __init__(self,
                 annotations,
                 target_sr=22050,
                 target_seconds = 6,
                 fps = 100,
                 train = True,
                 augmentations = None,
                 transform=None,
                 *args, **kwargs):
        self.annotations = annotations
        self.transform = transform
        self.target_sr = target_sr
        self.target_seconds = target_seconds
        self.hop_length = target_sr // fps
        self.fps = fps
        self.train = train
        self.augmentations = augmentations
        
        
    def __len__(self):
        return len(self.annotations)
    
    
    
    def __getitem__(self, index):
        
        
        x = self.annotations.iloc[index]
        path = x['file_path']
        beats = x['beats']
        downbeats = x['downbeats']

        # load the audio file
        # self.fps frames per second spectrogram.
        
        sr = x['sr']
        fps = x['fps']
        
        assert sr == self.target_sr
        assert fps == self.fps
        
        spectrogram = np.load(path)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        
        #if spectrogram is shorter than target_seconds * fps, pad it with zeros along the last dimension
        trunc= True
        if self.target_seconds is not None:
            if spectrogram.shape[-1] < self.target_seconds * self.fps :
                spectrogram = torch.nn.functional.pad(spectrogram, (0, self.target_seconds * self.fps - spectrogram.shape[-1]))
                trunc = False
        
        #beats and downbeats is a list of times in seconds. Convert to binary sequence the size of the spectrogram
        
        beat_sequence = np.zeros(spectrogram.shape[-1])
        downbeat_sequence = np.zeros(spectrogram.shape[-1])
        
        for beat in beats:
            beat_sequence[int(beat * self.fps)] = 1
        for downbeat in downbeats:
            downbeat_sequence[int(downbeat * self.fps)] = 1
            
        # apply a smoothing filter to the beat sequence with convoltion and a gaussian kernel
        beat_sequence = np.convolve(beat_sequence, np.array([0.25, 0.5, 1, 0.5, 0.25]), mode='same')
        downbeat_sequence = np.convolve(downbeat_sequence, np.array([0.25, 0.5, 1, 0.5, 0.25]), mode='same')
        
        
        beat_sequence = torch.tensor(beat_sequence, dtype=torch.float32)
        downbeat_sequence = torch.tensor(downbeat_sequence, dtype=torch.float32)
        
        if self.target_seconds is not None and trunc:
            start = np.random.randint(0, spectrogram.shape[-1] - self.target_seconds * self.fps)
            spectrogram = spectrogram[:,:, start:start + self.target_seconds * self.fps]
            beat_sequence = beat_sequence[start:start + self.target_seconds * self.fps]
            downbeat_sequence = downbeat_sequence[start:start + self.target_seconds * self.fps]
            
           
        # convert to tensor
        
        
        # return the audio and the label
        
        return {
            'spectrogram': spectrogram,
            'audio': torch.Tensor(0),
            'beats': beat_sequence,
            'downbeats': downbeat_sequence,
            'original_sample_rate': sr,
            'new_sample_rate': self.target_sr,
            'fps': self.fps,
        }