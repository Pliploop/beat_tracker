from beat_tracker.dataloading.datasets import BeatTrackingMelDataset, BeatTrackingDataset
from pytorch_lightning import LightningDataModule
import torch
import pandas as pd
import os
import numpy as np

class BeatTrackingDatamodule(LightningDataModule):
    
    def __init__(self, 
                 tasks = ['ballroom_mel'],
                 audio_dirs = {
                    'ballroom_mel':'/import/c4dm-datasets/ballroom/BallroomData',
                    'hainsworth_mel':'/import/c4dm-datasets/hainsworth',
                    'gtzan_mel':'/import/c4dm-datasets/gtzan_torchaudio'
                    },
                 target_sr=22050,
                 target_seconds = 6,
                 n_fft = 2048,
                 fps = 100,
                 n_mels = 128,
                 batch_size = 32,
                 val_split = 0.1,
                 test_split = 0.1,
                 num_workers = 0,
                 kfolds= None,
                 transform = False):
        super().__init__()
        
        
        self.tasks = tasks
        self.audio_dirs = audio_dirs
        self.target_sr = target_sr
        self.target_seconds = target_seconds
        self.n_fft = n_fft
        self.fps = fps
        self.n_mels = n_mels
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.k_fold = kfolds
        
        # overrides val_split and test_split if k_fold is not None
        if kfolds is not None:
            self.val_split = 1 / kfolds
            self.test_split = 1 / kfolds
        
        self.annotations =[]
        for task in tasks:
            fetch_annotations = getattr(self, f'fetch_{task}_annotations')
            annotations = fetch_annotations()
            self.annotations.append(annotations)
            
        self.annotations = pd.concat(self.annotations)
        
        # if k-fold is not None, split the annotations into k folds by creating k 'split' columns based on existing 'split' column
        # by rotating the indices of the annotations dataframe
        
        if kfolds is not None:
            samples_to_shift = len(self.annotations[self.annotations['split'] == 'val'])
            for i in range(0, kfolds):
                #the test set is always the same, k-fold is only for train and val
                
                # to_roll[f'split_{i}'] = pd.Series(np.roll(to_roll['split'],i*samples_to_shift))
                self.annotations[f'split_{i}'] = self.annotations['split']
                self.annotations.loc[
                    self.annotations['split'] != 'test', f'split_{i}'] = pd.Series(
                        np.roll(
                            self.annotations.loc[
                                self.annotations['split'] != 'test', f'split_{i}'
                                ].values,
                            i*samples_to_shift)
                        )
            self.annotations = self.annotations.dropna()
                
            
        
        self.transform = transform
        self.augmentations = None
        
        
    def fetch_ballroom_annotations(self):
        annotations_path = 'data/ballroom_annotations.json'
        
        annotations = pd.read_json(annotations_path)
        
        audio_dir = self.audio_dirs['ballroom_mel']
        
        for root, dirs, files in os.walk(audio_dir):
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
        annotations['task'] = 'ballroom'
        
        return annotations
    
    def fetch_gtzan_annotations(self):
        annotations_path = 'data/gtzan_annotations.json'
        
        annotations = pd.read_json(annotations_path)
        
        audio_dir = self.audio_dirs['gtzan_mel']
        
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    if file in annotations['file_name'].values:
                        annotations.loc[annotations['file_name'] == file, 'file_path'] = file_path
        # select train test and val indices
        annotations['split'] = 'test'
        annotations['task'] = 'gtzan'
        
        return annotations
    
    def fetch_hainsworth_annotations(self):
        annotations_path = 'data/hainsworth_annotations.json'
        
        audio_dir = self.audio_dirs['hainsworth_mel']
        
        annotations = pd.read_json(annotations_path)
        
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    if file in annotations['file_name'].values:
                        annotations.loc[annotations['file_name'] == file, 'file_path'] = file_path
                        
        # select train test and val indices
        annotations['split'] = 'train'
        annotations = annotations.sample(frac=1).reset_index(drop=True)
        n = len(annotations)
        val_idx = int(self.val_split * n)
        test_idx = int(self.test_split * n)
        annotations.loc[annotations.index[:val_idx], 'split'] = 'val'
        annotations.loc[annotations.index[val_idx:val_idx+test_idx], 'split'] = 'test'
        annotations['task'] = 'hainsworth'
        
        return annotations
    
    def fetch_ballroom_mel_annotations(self):
        annotations_path = 'data/ballroom_annotations_mel.json'
        
        annotations = pd.read_json(annotations_path)
        
        audio_dir = self.audio_dirs['ballroom_mel']
        
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    if file in annotations['file_name'].values:
                        annotations.loc[annotations['file_name'] == file, 'file_path'] = file_path
        # select train test and val indices
        annotations['split'] = 'train'
        annotations = annotations.sample(frac=1).reset_index(drop=True)
        n = len(annotations)
        val_idx = int(self.val_split * n)
        test_idx = int(self.test_split * n)
        annotations.loc[annotations.index[:val_idx], 'split'] = 'val'
        annotations.loc[annotations.index[val_idx:val_idx+test_idx], 'split'] = 'test'
        annotations['task'] = 'ballroom_mel'
        
        return annotations
    
    def setup(self, stage = None, fold = None):
        # get the split column according to the fold
        if fold is not None:
            self.fold = fold
            fold_col = f'split_{fold}'
        else:
            fold_col = 'split'
        
        train_annotations = self.annotations[self.annotations[fold_col] == 'train']
        val_annotations = self.annotations[self.annotations[fold_col] == 'val']
        test_annotations = self.annotations[self.annotations[fold_col] == 'test']
        
        if 'mel' not in self.tasks[0]:
            self.train_dataset = BeatTrackingDataset(train_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = True, augmentations = self.augmentations, transform = self.transform)
            self.val_dataset = BeatTrackingDataset(val_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = True, augmentations = self.augmentations, transform = self.transform)
            self.test_dataset = BeatTrackingDataset(test_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = False, augmentations = None, transform = False)  
        else:
            self.train_dataset = BeatTrackingMelDataset(train_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = True, augmentations = self.augmentations, transform = self.transform)
            self.val_dataset = BeatTrackingMelDataset(val_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = True, augmentations = self.augmentations, transform = self.transform)
            self.test_dataset = BeatTrackingMelDataset(test_annotations, target_sr=self.target_sr, target_seconds=self.target_seconds, n_fft=self.n_fft, fps=self.fps, n_mels=self.n_mels, train = False, augmentations = None, transform = False)
            
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = 1, shuffle = False, num_workers = self.num_workers)