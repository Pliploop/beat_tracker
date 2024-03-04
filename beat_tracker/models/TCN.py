
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCNLayer(nn.Module):
    
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=5,
            stride=1,
            padding=4,
            dropout=0.1):
        
        super(TCNLayer, self).__init__()

        self.conv1 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv1 = weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv2 = weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.elu3 = nn.ELU()

    def forward(self, x):
        
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.elu3(y)

        return y

class TCN(nn.Module):
    
    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):
       
        super(TCN, self).__init__()

        self.layers = []
        n_levels = len(channels)

        for i in range(n_levels):
            dilation = 2 ** i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.layers.append(
                TCNLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        
        y = self.net(x)
        return y
    


class BeatTrackingTCN(nn.Module):
    

    def __init__(
            self,
            channels=16,
            kernel_size=5,
            dropout=0.1):
        
        super(BeatTrackingTCN, self).__init__()

        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, channels, (3, 3), padding=(1, 0)),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 3))
        )

        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 0)),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 3))
        )

        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 8)),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        self.frontend = nn.Sequential(
            self.convblock1,
            self.convblock2,
            self.convblock3 
        )

        self.tcn = TCN(
            channels,
            [channels] * 11,
            kernel_size,
            dropout)
        
        self.out = nn.Conv1d(channels, 2, 1)
        
    def forward(self, spec):
        
        # if unbatched, add a batch dimension
        if len(spec.shape) == 3:
            spec = spec.unsqueeze(0)
            
        spec = spec.permute(0, 1, 3, 2) # (batch, channels, time, freq)
        
        frontend = self.frontend(spec)

        pre_tcn = frontend.squeeze(-1)
        tcn_out = self.tcn(pre_tcn)

        logits = self.out(tcn_out)

        return {
            "tcn_out": tcn_out,
            "logits": logits,
            "pre_tcn": pre_tcn
        }