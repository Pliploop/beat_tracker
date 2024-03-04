# lightningmodule for beat tracker

from pytorch_lightning.cli import OptimizerCallable
from pytorch_lightning import LightningModule
from torch import nn
import torch
from madmom.features import DBNBeatTrackingProcessor
from madmom.evaluation.beats import BeatEvaluation, BeatMeanEvaluation
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        self.fold = None
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
    
        
        self.test_agg = {
            'task': [],
            'preds': [],
            'targets': {
                'beats': [],
                'downbeats': []
            }
        }
        
        self.task2idx = {
            "ballroom_mel": 0,
            "hainsworth_mel": 1,
            "gtzan_mel": -1,
        }
        
        self.idx2task = {
            0: "ballroom_mel",
            1: "hainsworth_mel",
            -1: "gtzan_mel",
        }
        
        self.initial_state_dict = self.state_dict()
        
    def forward(self, x):
        return self.model(x)
    
    def single_times_from_activations(self, beat_logits, downbeat_logits = None):
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
            'cemgil_beats': beat_score.cemgil,
            'cemgil_downbeats': downbeat_score.cemgil,
            'cmlc_beats': beat_score.cmlc,
            'cmlc_downbeats': downbeat_score.cmlc,
            'cmlt_beats': beat_score.cmlt,
            'cmlt_downbeats': downbeat_score.cmlt,
            'amlc_beats': beat_score.amlc,
            'amlc_downbeats': downbeat_score.amlc,
            'amlt_beats': beat_score.amlt,
            'amlt_downbeats': downbeat_score.amlt,
            'information_gain_beats': beat_score.information_gain,
            'information_gain_downbeats': downbeat_score.information_gain}
        
        return {
            'averages': metrics,
            'beat_scores': beat_scores,
            'downbeat_scores': downbeat_scores
        }
        
        
    
    def update_fold(self, fold):
        """
        Update the fold value and reset the optimizer, scheduler, and model weights.
        
        Parameters:
        fold (int): The new fold value.
        """
        
        self.fold = fold
        if fold != 0 and fold is not None:
            # reset optimizers, schedulers and model weights
            self.load_state_dict(self.initial_state_dict)
            # reset the optimizer and scheduler using the same hyperparams but instantiate new objects
            # get the optimizer object
            self.optimizers().load_state_dict(self.optim_state)
            
            self.scheduler.load_state_dict(self.scheduler_state)
        
        
    
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
        # log lr
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        if self.fold is not None:
            # log all the losses with fold in the name such as train_fold_k_loss
            self.log(f'train_fold_{self.fold}_loss', total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'train_fold_{self.fold}_beat_loss', beat_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'train_fold_{self.fold}_downbeat_loss', downbeat_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss
    
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
        
        if self.fold is not None:
            # log all the losses with fold in the name such as val_fold_k_loss
            self.log(f'val_fold_{self.fold}_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'val_fold_{self.fold}_beat_loss', beat_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f'val_fold_{self.fold}_downbeat_loss', downbeat_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss
    
        
    def test_step(self, batch, batch_idx):
            """
            Perform a single step of testing on a batch of data.

            Args:
                batch (dict): A dictionary containing the batch data.
                batch_idx (int): The index of the current batch.

            Returns:
                None
            """
            
            spec = batch['spectrogram']
            beats = batch['beats']
            downbeats = batch['downbeats']
            task = batch['task'].item()
            task = self.idx2task[task]
            
            y = self.model(spec)
            
            self.test_agg['task'].append(task)
            self.test_agg['preds'].append(y['logits'].detach().cpu())
            self.test_agg['targets']['beats'].append(beats.detach().cpu())
            self.test_agg['targets']['downbeats'].append(downbeats.detach().cpu())
        
        
    def on_test_epoch_end(self):
        """
        Performs operations at the end of each test epoch.
        """
        self.test_agg['preds'] = torch.cat(self.test_agg['preds'])
        self.test_agg['targets']['beats'] = torch.cat(self.test_agg['targets']['beats'])
        self.test_agg['targets']['downbeats'] = torch.cat(self.test_agg['targets']['downbeats'])
        
        metrics = self.get_metrics(self.test_agg)
        if metrics is not None:
            for k,v in metrics['averages'].items():
                self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        beat_scores = metrics['beat_scores']
        downbeat_scores = metrics['downbeat_scores']
        
        # create a dataframe with the results and the task
        results = {
            'task': self.test_agg['task'],
            'f_measure_beats': [x.fmeasure for x in beat_scores],
            'f_measure_downbeats': [x.fmeasure for x in downbeat_scores],
            'cemgil_beats': [x.cemgil for x in beat_scores],
            'cemgil_downbeats': [x.cemgil for x in downbeat_scores],
            'cmlc_beats': [x.cmlc for x in beat_scores],
            'cmlc_downbeats': [x.cmlc for x in downbeat_scores],
            'cmlt_beats': [x.cmlt for x in beat_scores],
            'cmlt_downbeats': [x.cmlt for x in downbeat_scores],
            'amlc_beats': [x.amlc for x in beat_scores],
            'amlc_downbeats': [x.amlc for x in downbeat_scores],
            'amlt_beats': [x.amlt for x in beat_scores],
            'amlt_downbeats': [x.amlt for x in downbeat_scores],
            'information_gain_beats': [x.information_gain for x in beat_scores],
            'information_gain_downbeats': [x.information_gain for x in downbeat_scores]
        }
        
        dataframe = pd.DataFrame(results)
        print(dataframe.groupby('task').mean().T)
        
        grouped = dataframe.groupby('task',as_index=False).mean()
        
        if self.logger:
            self.logger.log_table(key='test_metrics', columns=grouped.columns.tolist(), data=grouped.values.tolist())
        
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
            
        self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True,min_lr=1e-5)
        
        # save the optimizer and scheduler original states
        self.optim_state = optimizer.state_dict()
        self.scheduler_state = self.scheduler.state_dict()
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def load_weights(self, path):
        self.load_state_dict(torch.load(path)['state_dict'])