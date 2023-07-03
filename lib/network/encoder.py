import torch
from torch import nn
from .resnet_block import ResNetBlock
from .attention import make_attn
from .utils import Downsample, normalize, nonlinearity


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 in_resolution,
                 res_block_num,
                 attn_resolutions,
                 z_channels,  # ?
                 double_z=True,
                 channel_series=(1,2,4,8),
                 dropout=0.,
                 attn_type='vanilla',
                 resample_with_conv=True):
        super().__init__()
        
        self.resolution_num = len(channel_series)
        self.res_block_num = res_block_num
        self.mid_channels = mid_channels
        self.time_embed = 0
        
        self.conv_in = nn.Conv2d(in_channels, 
                                 self.mid_channels, 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=1)
        
        in_channel_series = (1,) + channel_series[:-1]
        current_resolution = in_resolution
        self.down = nn.ModuleList()
        for level in range(self.resolution_num):
            res_block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.mid_channels * in_channel_series[level]
            block_out = self.mid_channels * channel_series[level]
            
            for _ in range(self.res_block_num):
                res_block.append(ResNetBlock(in_channels=block_in, 
                                             out_channels=block_out, 
                                             time_embed_channels=self.time_embed, 
                                             dropout=dropout))
                block_in = block_out
                if current_resolution in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
                    
            down_sample = nn.Module()
            down_sample.block = res_block
            down_sample.attn = attn
            
            if level != self.resolution_num - 1:
                down_sample.downsample = Downsample(block_in, resample_with_conv)
                current_resolution = current_resolution // 2
                
            self.down.append(down_sample)
            
        self.mid = nn.Module()
        self.mid.block1 = ResNetBlock(in_channels=block_in,
                                      out_channels=block_in,
                                      time_embed_channels=self.time_embed,
                                      dropout=dropout)
        self.mid.attn1 = make_attn(block_in, attn_type=attn_type)
        
        self.mid.block2 = ResNetBlock(in_channels=block_in,
                                      out_channels=block_in,
                                      time_embed_channels=self.time_embed,
                                      dropout=dropout)
        
        self.normout = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  2*z_channels if double_z else z_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
    
    def forward(self, x):
        time_embed = None
        
        # ! different from og
        h = self.conv_in(x)
        for level in range(self.resolution_num):
            for block in range(self.res_block_num):
                h = self.down[level].block[block](h, time_embed)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[block](h)
            if level != self.resolution_num - 1:
                h = self.down[level].downsample(h)
        
        h = self.mid.block1(h, time_embed)
        h = self.mid.attn1(h)
        h = self.mid.block2(h, time_embed)
        
        h = self.norm(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h
