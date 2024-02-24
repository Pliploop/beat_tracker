from beat_tracker.dataloading.datasets import BeatTrackingDataset
from pytorch_lightning import LightningDataModule
import torch
import pandas as pd
import os

class BeatTrackingDatamodule(LightningDataModule):
    
    def __init__(self, 
                 tasks = ['ballroom'],
                 audio_dir = '/import/c4dm-datasets/ballroom/BallroomData',
                 target_sr=22050,
                 target_seconds = 6,
                 n_fft = 2048,
                 fps = 100,
                 n_mels = 128,
                 batch_size = 32,
                 val_split = 0.1,
                 test_split = 0.1,
                 num_workers = 0,
                 transform = False):
        super().__init__()
        
        
        self.tasks = tasks
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.target_seconds = target_seconds
        self.n_fft = n_fft
        self.fps = fps
        self.n_mels = n_mels
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        
        self.annotations =[]
        for task in tasks:
            fetch_annotations = getattr(self, f'fetch_{task}_annotations')
            annotations = fetch_annotations()
            self.annotations.append(annotations)
            
        self.annotations = pd.concat(self.annotations)
        
        self.transform = transform
        self.augmentations = None
        
        self.audio_dir = audio_dir
        
    def fetch_ballroom_annotations(self):
        annotations_path = 'data/ballroom_annotations.csv'
        
        annotations = pd.read_csv(annotations_path)
        
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    if file in annotations['file_name'].values:
                        annotations.loc[annotations['file_name'] == file, 'file_path'] = file_path
        # select train test and val indices
        annotations['split'] = 'train'
        annotations = annotations.sample(frac=1).reset_index(drop=True) #shuffle
        n = len(annotations)
        val_idx = int(self.val_split * n)
        test_idx = int(self.test_split * n)
        annotations.loc[annotations.index[:val_idx], 'split'] = 'val'
        annotations.loc[annotations.index[val_idx:val_idx+test_idx], 'split'] = 'test'
        
        return annotations
    
    def setup(self, stage = None):
        train_annotations = self.annotations[self.annotations['split'] == 'train']
        val_annotations = self.annotations[self.annotations['split'] == 'val']
        test_annotations = self.annotations[self.annotations['split'] == 'test']
        self.train_dataset = BeatTrackingDataset(train_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = True, augmentations = self.augmentations, transform = self.transform)
        self.val_dataset = BeatTrackingDataset(val_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = True, augmentations = self.augmentations, transform = self.transform)
        self.test_dataset = BeatTrackingDataset(test_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = False, augmentations = None, transform = False)  
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = 1, shuffle = False, num_workers = self.num_workers)