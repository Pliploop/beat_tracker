"""
This script trains a BeatTracker model using PyTorch Lightning.
"""

from beat_tracker.models.beat_tracker import BeatTracker
from beat_tracker.dataloading.datamodule import BeatTrackingDatamodule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import yaml
import os

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Saves the configuration of the experiment.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
            pl_module (LightningModule): The PyTorch Lightning LightningModule object.
            stage (str): The current stage of training.

        Returns:
            None
        """
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                trainer.logger.experiment.config.update(config, allow_val_change=True)
            with open(os.path.join(os.path.join(self.config['ckpt_path'], experiment_name), "config.yaml"), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
                
            #instanciate a ModelCheckpoint saving the model every epoch
            
            


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        Adds custom arguments to the parser.

        Args:
            parser (ArgumentParser): The argument parser.

        Returns:
            None
        """
        parser.add_argument("--log", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints")
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--test", default=False)
        parser.add_argument("--early_stopping_patience", default=10)


if __name__ == "__main__":

    cli = MyLightningCLI(model_class=BeatTracker, datamodule_class=BeatTrackingDatamodule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},)
    
    cli.instantiate_classes()

    if cli.config.log:
        logger = WandbLogger(project="BeatTracker", id=cli.config.resume_id)

        experiment_name = logger.experiment.name
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None

    cli.trainer.logger = logger
    
    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)) and not cli.config.test:
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    fold = None
    # get the metrics from the fold
    
    # initial callbacks
    # two model checkpoints that save the best model based on the validation loss and the best model based on the training loss
    if logger is None:
        experiment_name = 'test'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
        filename='best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback_train = ModelCheckpoint(
        dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
        filename='best-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=cli.config.early_stopping_patience,
            mode='min'
        )

    callbacks = [checkpoint_callback, checkpoint_callback_train, early_stopping_callback]


    cli.trainer.callbacks = cli.trainer.callbacks[:-1]+callbacks

    
    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)):
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    
    if cli.config.data.kfolds is not None:
        
        cli.trainer.callbacks = cli.trainer.callbacks[:-2]
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
            filename=f'best-{{epoch:02d}}-{{val_fold_loss:.2f}}',
            save_top_k=1,
            monitor=f'val_loss',
            mode='min'
        )
        
        checkpoint_callback_train = ModelCheckpoint(
            dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
            filename=f'best-{{epoch:02d}}-{{train_loss:.2f}}',
            save_top_k=1,
            monitor=f'train_loss',
            mode='min'
        )
        
        early_stopping_callback = EarlyStopping(
                monitor=f'val_fold_{fold}_loss',
                patience=cli.config.early_stopping_patience,
                mode='min'
            )
            
        callbacks = [checkpoint_callback, checkpoint_callback_train, early_stopping_callback]
        cli.trainer.callbacks = cli.trainer.callbacks+callbacks
         
        for fold in range(cli.config.data.kfolds):
            
            
            print(f"Training fold {fold}")
            
            new_early_stopping_callback = EarlyStopping(
                monitor=f'val_fold_{fold}_loss',
                patience=cli.config.early_stopping_patience,
                mode='min'
            )
            
            cli.trainer.callbacks = cli.trainer.callbacks[:-1]+[new_early_stopping_callback]        
            
            cli.model.update_fold(fold)
            cli.datamodule.setup(fold=fold)
            
            # create a new trainer with the same parameters
            kwargs = cli.config.trainer
            new_trainer = Trainer(**kwargs)
            new_trainer.logger = logger
            new_trainer.callbacks = cli.trainer.callbacks
            new_trainer.fit(model=cli.model, datamodule=cli.datamodule)
            
            
            
    else:
        pass
    
    if not cli.config.test:
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.resume_from_checkpoint)
        if logger is not None:
                cli.model.load_weights(checkpoint_callback.best_model_path)
                
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
    
