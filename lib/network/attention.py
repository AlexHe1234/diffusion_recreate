import torch
from torch import nn
from .utils import normalize
import torch.nn.functional as F
from einops import rearrange, repeat


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        h = x
        h = self.norm(h)
        
        q = self.q(h)
        k = self.k(h)
        v = self.v(v)
        
        B, C, H, W = q.shape
        q = q.reshape(B, C, H*W)
        q = q.permute(0, 2, 1)
        k = k.reshape(B, C, H*W)
        w = torch.bmm(q, k)
        w = w * (int(C) ** -0.5)
        w = F.softmax(w, dim=2)
        v = v.reshape(B, C, H*W)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        h = h.reshape(B, C, H, W)
        
        h = self.proj_out(h)
        return x + h
    
    
class LinearAttn(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (q_k_v heads c) h w -> q_k_v b heads c (h w)', heads=self.heads, q_k_v=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=H, w=W)
        return self.to_out(out)
    

def make_attn(in_channels, attn_type='vanilla'):
    assert attn_type in ['vanilla', 'linear', 'none'], 'attention type unknown'
    if attn_type == 'vanilla':
        return AttnBlock(in_channels)
    elif attn_type == 'linear':
        return LinearAttn(dim=in_channels, heads=1, dim_head=in_channels)
    else:
        return nn.Identity(in_channels)
