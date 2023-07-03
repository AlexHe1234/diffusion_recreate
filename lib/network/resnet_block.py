import torch
from torch import nn
from .utils import normalize, nonlinearity


class ResNetBlock(nn.Module):
    def __init__(self, 
                 dropout, 
                 in_channels, 
                 out_channels=None, 
                 conv_shortcut=False, 
                 time_embed_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = normalize(self.in_channels)
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        
        # used in ldm
        if time_embed_channels > 0:
            self.time_embed_proj = nn.Linear(time_embed_channels, out_channels)
            
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, 
                                               out_channels, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, 
                                              out_channels, 
                                              kernel_size=1, 
                                              stride=1, 
                                              padding=0)
                
    def forward(self, x, time_embed):
        h = x
        
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        if time_embed is not None:
            h = h + self.time_embed_proj(nonlinearity(time_embed))[:, :, None, None]
            
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
                
        return x + h
