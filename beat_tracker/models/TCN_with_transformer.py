from torch import nn
from beat_tracker.models.TCN import BeatTrackingTCN
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import torch
import numpy as np

 

class BeatTrackingTCNTransformer(nn.Module):
    
    def __init__(
            self,
            channels=16,
            kernel_size=5,
            dropout=0.1,
            downbeats=True,
            nhead = 4,
            num_layers = 2):
        
        super(BeatTrackingTCNTransformer, self).__init__()

        self.tcn = BeatTrackingTCN(
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,downbeats=True)
        
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=channels, nhead=nhead, batch_first=True)
        self.beats_decoder = nn.TransformerDecoder(self.transformer_layer, num_layers=num_layers)
        self.downbeats_decoder = nn.TransformerDecoder(self.transformer_layer, num_layers=num_layers)
        
        self.transformer_sequence_length = 600 # 6 seconds at 100 fps
        
        
        
        #implement rotary position encodings
        
    def forward(self, x):
        
        y = self.tcn(x)
        
        logits = y['logits']
        
        beats = logits[:,0,:] # batch,time
        downbeats = logits[:,1,:] # batch,time
        
        # the decoders slide over the beats and downbeats in the time dimension -1
        
        # beats = torch.split(beats, self.transformer_sequence_length, dim=-1)
        # downbeats = torch.split(downbeats, self.transformer_sequence_length, dim=-1)
        
        # # pad the last sequence with zeros after creating a padding mask
        # padding_mask = torch.nn.functional.pad(torch.ones_like(beats[-1]), (0, self.transformer_sequence_length - beats[-1].shape[-1]))
        # beats = torch.nn.functional.pad(beats, (0, self.transformer_sequence_length - beats[-1].shape[-1]))
        # downbeats = torch.nn.functional.pad(downbeats, (0, self.transformer_sequence_length - downbeats[-1].shape[-1]))
        
        # instead of doing the above, create an attention mask where tokens can only pay attention to tokens 300 timesteps in the past and the future
        # and use relative position encodings
        
        attention_mask = torch.zeros(beats.shape[-1], beats.shape[-1])
        # binary mask where 1s are the tokens that can be attended to
        for i in range(beats.shape[-1]):
            attention_mask[i, max(0,i-300):min(beats.shape[-1],i+300)] = 1
        
        
        beats = self.beats_decoder(tgt = beats, memory = downbeats, tgt_key_padding_mask = padding_mask, memory_key_padding_mask = padding_mask)
        downbeats = self.downbeats_decoder(tgt = downbeats, memory = beats, tgt_key_padding_mask = padding_mask, memory_key_padding_mask = padding_mask)
        
        # concatenate the beats and downbeats back together as well as the padding masks
        beats = torch.cat(beats, dim=-1)
        downbeats = torch.cat(downbeats, dim=-1)
        
        out_ = torch.cat([beats, downbeats], dim=1)
        
        # truncate y to the original length
        out_ = out_[:,:-padding_mask.sum().item(),:]
        
        return y
    
    