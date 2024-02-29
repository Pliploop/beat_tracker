# lightningmodule for beat tracker


from beat_tracker.models.TCN_with_transformer import BeatTrackingTCNTransformer
from beat_tracker.models.TCN import BeatTrackingTCN
from beat_tracker.utils.metrics import f_measure, continuity
from pytorch_lightning.cli import OptimizerCallable
from pytorch_lightning import LightningModule
from torch import nn
import torch
import numpy as np
from madmom.features import DBNBeatTrackingProcessor
from madmom.evaluation.beats import BeatEvaluation, BeatMeanEvaluation
from tqdm import tqdm



class BeatTracker(LightningModule):
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: OptimizerCallable = None,
                 scheduler = None,
                 beat_vs_downbeat_loss_ratio = 0.5,
                 threshold = 0.5,
                 fps = 100,
                 *args, **kwargs):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.beat_vs_downbeat_loss_ratio = beat_vs_downbeat_loss_ratio
        
        self.agg_preds = {}
        self.agg_targets = {}
        self.fold = 0
        self.threshold = threshold
        self.fps = fps
        
        self.beat_postprocessor =  DBNBeatTrackingProcessor(
            min_bpm=55,
            max_bpm=215,
            transition_lambda=100,
            fps=self.fps,
            online=True)
        
        self.downbeat_postprocessor = DBNBeatTrackingProcessor(
            min_bpm=10,
            max_bpm=75,
            transition_lambda=100,
            fps=self.fps,
            online=True)
        
        
        self.train_agg = {
            'preds': [],
            'targets': {
                'beats': [],
                'downbeats': []
            }
        }
        
        self.val_agg = {
            'preds': [],
            'targets': {
                'beats': [],
                'downbeats': []
            }
        }
        
        self.test_agg = {
            'preds': [],
            'targets': {
                'beats': [],
                'downbeats': []
            }
        }
        
    def forward(self, x):
        return self.model(x)
    
    def single_times_from_activations(self, beat_logits, downbeat_logits = None):
        print('x')
        beat_sigmoid = torch.sigmoid(beat_logits)
        downbeat_sigmoid = torch.sigmoid(downbeat_logits)
        self.beat_postprocessor.reset()
        self.downbeat_postprocessor.reset()
        beat_times = self.beat_postprocessor.process_offline(beat_sigmoid.squeeze())
        downbeat_times = self.downbeat_postprocessor.process_offline(downbeat_sigmoid.squeeze())
            
        return beat_times, downbeat_times
    
    def single_times_from_binary(self, beat_logits, downbeat_logits = None):
        beat_times = torch.where(beat_logits > 0.5)[0]/self.fps
        downbeat_times = torch.where(downbeat_logits > 0.5)[0]/self.fps
        return beat_times.detach().cpu().numpy(), downbeat_times.detach().cpu().numpy()
        
    def batch_times_from_activations(self, beat_logits, downbeat_logits = None):
        beat_times = []
        downbeat_times = []
        for i in range(len(beat_logits)):
            beat_time, downbeat_time = self.single_times_from_activations(beat_logits[i], downbeat_logits[i])
            beat_times.append(beat_time)
            downbeat_times.append(downbeat_time)
        return beat_times, downbeat_times
    
    def batch_times_from_binary(self, beat_logits, downbeat_logits = None):
        beat_times = []
        downbeat_times = []
        for i in range(len(beat_logits)):
            beat_time, downbeat_time = self.single_times_from_binary(beat_logits[i], downbeat_logits[i])
            beat_times.append(beat_time)
            downbeat_times.append(downbeat_time)
        return beat_times, downbeat_times
    
    def get_metrics(self, agg):
        
        beat_logits = agg['preds'][:,0,:]
        downbeat_logits = agg['preds'][:,1,:]
        gt_beats = agg['targets']['beats']
        gt_downbeats = agg['targets']['downbeats']
        
        est_beat_times, est_downbeat_times = self.batch_times_from_activations(beat_logits, downbeat_logits)
        gt_beat_times, gt_downbeat_times = self.batch_times_from_binary(gt_beats, gt_downbeats)
        
        beat_scores = []
        downbeat_scores = []
        for i in tqdm(range(len(est_beat_times))):
            beat_score = BeatEvaluation(est_beat_times[i], gt_beat_times[i])
            downbeat_score = BeatEvaluation(est_downbeat_times[i], gt_downbeat_times[i])
            beat_scores.append(beat_score)
            downbeat_scores.append(downbeat_score)
            
        beat_score = BeatMeanEvaluation(beat_scores)
        downbeat_score = BeatMeanEvaluation(downbeat_scores)
        
        
        metrics = {
            'f_measure_beats': beat_score.fmeasure,
            'f_measure_downbeats': downbeat_score.fmeasure,
            # 'CMLc_beat': CMLc_beat,
            # 'CMLt_beat': CMLt_beat,
            # 'AMLc_beat': AMLc_beat,
            # 'AMLt_beat': AMLt_beat,
            # 'CMLc_downbeat': CMLc_downbeat,
            # 'CMLt_downbeat': CMLt_downbeat,
            # 'AMLc_downbeat': AMLc_downbeat,
            # 'AMLt_downbeat': AMLt_downbeat
        }
        
        return metrics
        
        
    
    def update_fold(self, fold):
        self.fold = fold
    
    def training_step(self, batch, batch_idx):
        spec = batch['spectrogram']
        beats = batch['beats']
        downbeats = batch['downbeats']
        
        y = self.model(spec)
        
        beat_loss = self.criterion(y['logits'][:,0,:], beats)
        downbeat_loss = self.criterion(y['logits'][:,1,:], downbeats)
        
        total_loss = beat_loss + downbeat_loss
    
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_beat_loss', beat_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_downbeat_loss', downbeat_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        # metrics = self.get_metrics(y, batch)
        # if metrics is not None:
        #     for k,v in metrics.items():
        #         self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.train_agg['preds'].append(y['logits'].detach().cpu())
        self.train_agg['targets']['beats'].append(beats.detach().cpu())
        self.train_agg['targets']['downbeats'].append(downbeats.detach().cpu())
        
        return total_loss
    
    def on_train_epoch_end(self):
        self.train_agg['preds'] = torch.cat(self.train_agg['preds'])
        self.train_agg['targets']['beats'] = torch.cat(self.train_agg['targets']['beats'])
        self.train_agg['targets']['downbeats'] = torch.cat(self.train_agg['targets']['downbeats'])
        
        # metrics = self.get_metrics(self.train_agg)
        # if metrics is not None:
        #     for k,v in metrics.items():
        #         self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
        self.train_agg = {
            'preds': [],
            'targets': {
                'beats': [],
                'downbeats': []
            }
        }
    
    def validation_step(self, batch, batch_idx):
        
        spec = batch['spectrogram']
        beats = batch['beats']
        downbeats = batch['downbeats']
        
        y = self.model(spec)
        
        beat_loss = self.criterion(y['logits'][:,0,:], beats)
        downbeat_loss = self.criterion(y['logits'][:,1,:], downbeats)
        
        total_loss = self.beat_vs_downbeat_loss_ratio * beat_loss + (1 - self.beat_vs_downbeat_loss_ratio) * downbeat_loss
    
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_beat_loss', beat_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_downbeat_loss', downbeat_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # metrics = self.get_metrics(y, batch)
        # if metrics is not None:
        #     for k,v in metrics.items():
        #         self.log(f'val_{k}', v, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_agg['preds'].append(y['logits'].detach().cpu())
        self.val_agg['targets']['beats'].append(beats.detach().cpu())
        self.val_agg['targets']['downbeats'].append(downbeats.detach().cpu())
        
        return total_loss
    
    def on_validation_epoch_end(self):
        
        self.val_agg['preds'] = torch.cat(self.val_agg['preds'])
        self.val_agg['targets']['beats'] = torch.cat(self.val_agg['targets']['beats'])
        self.val_agg['targets']['downbeats'] = torch.cat(self.val_agg['targets']['downbeats'])
        
        metrics = self.get_metrics(self.val_agg)
        if metrics is not None:
            for k,v in metrics.items():
                self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
        self.val_agg = {
            'preds': [],
            'targets': {
                'beats': [],
                'downbeats': []
            }
        }
    
        
    def test_step(self,batch,batch_idx):
        
        spec = batch['spectrogram']
        beats = batch['beats']
        downbeats = batch['downbeats']
        
        y = self.model(spec)
        
        # collate predictions and targets with self.agg_preds and self.agg_targets, all to cpu
        
        # metrics = self.get_metrics(y, batch)
        # if metrics is not None:
        #     for k,v in metrics.items():
        #         self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        
        self.test_agg['preds'].append(y['logits'].detach().cpu())
        self.test_agg['targets']['beats'].append(beats.detach().cpu())
        self.test_agg['targets']['downbeats'].append(downbeats.detach().cpu())  
        
        
    def on_test_epoch_end(self):
            
            self.test_agg['preds'] = torch.cat(self.test_agg['preds'])
            self.test_agg['targets']['beats'] = torch.cat(self.test_agg['targets']['beats'])
            self.test_agg['targets']['downbeats'] = torch.cat(self.test_agg['targets']['downbeats'])
            
            metrics = self.get_metrics(self.test_agg)
            if metrics is not None:
                for k,v in metrics.items():
                    self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                    
            self.test_agg = {
                'preds': [],
                'targets': {
                    'beats': [],
                    'downbeats': []
                }
            }  
    
    def configure_optimizers(self):
        
        if self.optimizer is not None:
            optimizer = self.optimizer(self.parameters())
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
            
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def load_weights(self, path):
        self.load_state_dict(torch.load(path)['state_dict'])