
from beat_tracker.models.TCN import BeatTrackingTCN
from madmom.features import DBNBeatTrackingProcessor
import torch
from beat_tracker.dataloading.loading_utils import get_spectrogram,load_audio
import yaml
import librosa

DEFAULT_CHECKPOINT = 'default_checkpoints/best-epoch=679-val_loss=0.10.ckpt'
DEFAULT_CONFIG = 'default_checkpoints/config.yaml'

class BeatTracker:
    """
    Class for beat tracking in audio files.

    Args:
        checkpoint_path (str): The path to the model checkpoint file.
        model_class (class): The class of the beat tracking model.
        config (str): The path to the configuration file.
        predict_downbeats (bool): Whether to predict downbeats in addition to beat times.

    Attributes:
        device (str): The device used for computation (cuda or cpu).
        checkpoint_path (str): The path to the model checkpoint file.
        model (object): The beat tracking model.
        predict_downbeats (bool): Whether to predict downbeats in addition to beat times.
        config (dict): The configuration settings.
        target_sample_rate (int): The target sample rate for audio processing.
        n_mels (int): The number of mel frequency bins.
        fps (int): The frames per second for beat tracking.
        hop_length (int): The hop length for audio processing.
        beat_postprocessor (object): The beat post-processing object.
        downbeat_postprocessor (object): The downbeat post-processing object.
        
    Example usage:
        >>> beat_tracker = BeatTracker()
        >>> beat_times, downbeat_times = beat_tracker('path/to/audio/file')
    """

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT,
                 model_class=BeatTrackingTCN, # this should also be taken from config
                 config=DEFAULT_CONFIG,
                 predict_downbeats=False) -> None:
        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_path = checkpoint_path
        self.model = model_class()
        self.predict_downbeats = predict_downbeats

        # load config as yaml
        with open(config, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.target_sample_rate = self.config['data']['target_sr']
        self.n_mels = self.config['data']['n_mels']
        self.fps = self.config['data']['fps']
        self.hop_length = self.target_sample_rate // self.fps

        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        if "model." == list(state_dict.keys())[0][:6]:
            new_state_dict = self.process_torch_state_dict(state_dict)
        self.model.load_state_dict(new_state_dict)

        self.beat_postprocessor = DBNBeatTrackingProcessor(
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

    def process_torch_state_dict(self, state_dict):
        """
        Process the given PyTorch state dictionary by removing the "model." prefix from the keys.
        This is to deal with pytorch lightning checkpoints without having to install pytorch lightning.

        Args:
            state_dict (dict): The PyTorch state dictionary to be processed.

        Returns:
            dict: The processed state dictionary with the "model." prefix removed from the keys.
        """
        new_state_dict = {}
        for key in state_dict:
            new_state_dict[key[6:]] = state_dict[key]

        return new_state_dict

    def __call__(self, path, return_audio = False):
        """
        Process an audio file and return the beat and downbeat times.

        Parameters:
        path (str): The path to the audio file.

        Returns:
        tuple: A tuple containing the beat times and downbeat times (if available).
        """
        audio, original_sr = load_audio(path, target_sr=self.target_sample_rate)
        gram = get_spectrogram(audio, target_sr=self.target_sample_rate, n_mels=self.n_mels,
                               hop_length=self.hop_length, n_fft=2048)
        activations = self.model(gram.unsqueeze(0).to(self.device))['logits']
        beat_activations = activations[0, 0, :]
        downbeat_activations = activations[0, 1, :]
        beat_times = self.beat_postprocessor(beat_activations.squeeze().cpu().detach().numpy())
        downbeat_times = None
        if self.predict_downbeats:
            downbeat_times = self.downbeat_postprocessor(downbeat_activations.squeeze().cpu().detach().numpy())
        if return_audio:
            return beat_times, downbeat_times, {
                "audio": audio,
                "spectrogram": gram,
                "sr": self.target_sample_rate
            }
        return beat_times, downbeat_times
        
        
    def sonify_beats(self, path, beat_times, downbeat_times=None):
        """
        Sonify the beat and downbeat times of an audio file.

        Parameters:
        path (str): The path to the audio file.
        beat_times (list): A list of beat times in seconds.
        downbeat_times (list): A list of downbeat times in seconds.

        Returns:
        numpy.ndarray: The audio with the beat and downbeat sounds added.
        """
        
        beats, downbeats, audio_dic = self(path, return_audio=True)
        audio = audio_dic["audio"]
        sr = audio_dic["sr"]
        gram = audio_dic["spectrogram"]
        
        # sonify the beats
        beat_audio = librosa.clicks(times=beat_times, sr=sr, length=len(audio), click_freq=500, click_duration=0.1)
        if downbeat_times is not None:
            downbeat_audio = librosa.clicks(times=downbeat_times, sr=sr, length=len(audio))
            beat_audio += downbeat_audio
            
        return audio + beat_audio
        
        
        
        
    
        