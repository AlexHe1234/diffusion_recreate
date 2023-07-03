import torch
from torch import nn
import torch.nn.functional as F


def normalize(in_channels, num_groups=32):
    # divide channels into groups and apply BN independently to each group
    # affine means the result is transformed w * x + b
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, 
                                  in_channels, 
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)
        
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x
