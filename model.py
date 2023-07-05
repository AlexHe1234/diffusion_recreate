import torch
from torch import nn
import torch.nn.functional as F
from config import cfg
import numpy as np


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.residual:
            if self.in_channels == self.out_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            return x2
        

class EmbedLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, x):
        x = x.view(-1, self.in_dim)
        return self.layer(x)
    
    
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels),
        )
        
    def forward(self, x, skip):
        x = torch.cat([x, skip], 1)
        x = self.model(x)
        return x


class UNet(nn.Module):
    def __init__(self, 
                in_channels,
                mid_channels=256,
                num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes
        
        self.conv0 = ResBlock(in_channels, mid_channels, residual=True)
        
        self.down1 = nn.Sequential(
            ResBlock(mid_channels, mid_channels),
            nn.MaxPool2d(2),
        )
        self.down2 = nn.Sequential(
            ResBlock(mid_channels, mid_channels*2),
            nn.MaxPool2d(2),
        )
        
        # (feat_channel,)
        self.to_vec = nn.Sequential(
            nn.AvgPool2d(7),
            nn.GELU(),
        )
        
        self.time_embed1 = EmbedLayer(1, 2*mid_channels)
        self.time_embed2 = EmbedLayer(1, mid_channels)
        
        self.context_embed1 = EmbedLayer(num_classes, 2*mid_channels)
        self.context_embed2 = EmbedLayer(num_classes, mid_channels)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*mid_channels, 2*mid_channels, 7, 7),
            nn.GroupNorm(8, 2*mid_channels),
            nn.ReLU(),
        )
        self.up1 = UNetUp(4*mid_channels, mid_channels)
        self.up2 = UNetUp(2*mid_channels, mid_channels)
        
        self.out = nn.Sequential(
            nn.Conv2d(2*mid_channels, mid_channels, 3, 1, 1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1),
        )
        
    def forward(self, x, c, t, context_mask):
        
        x = self.conv0(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        vec = self.to_vec(down2)
        
        c = F.one_hot(c, self.num_classes).type(torch.float)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.num_classes)  # batch * num_classes
        c = c * (context_mask - 1)
        
        c_embed1 = self.context_embed1(c).view(-1, self.mid_channels*2, 1, 1)
        c_embed2 = self.context_embed2(c).view(-1, self.mid_channels, 1, 1)
        
        t_embed1 = self.time_embed1(t).view(-1, self.mid_channels*2, 1, 1)
        t_embed2 = self.time_embed2(t).view(-1, self.mid_channels, 1, 1)
        up1 = self.up0(vec)
        up2 = self.up1(c_embed1*up1+t_embed1, down2)
        up3 = self.up2(c_embed2*up2+t_embed2, down1)
        return self.out(torch.cat([up3, x], 1))


class DDPM(nn.Module):
    def __init__(self,
                 beta_start,
                 beta_end,
                 device,
                 drop_prob=0.1
                 ):
        super().__init__()
        self.model = UNet(cfg.in_channels, cfg.mid_channels, cfg.num_classes)
        self.timestep = cfg.timestep
        self.drop_prob = drop_prob
        self.loss = nn.MSELoss()
        self.device = device
        for k, v in self.schedule(beta_start, beta_end, self.timestep).items():
            self.register_buffer(k, v)
        
    @staticmethod    
    def schedule(beta_start, beta_end, timestep):
        assert beta_start < beta_end < 1.0, \
            'starting and ending value of beta must be less than 1'
        beta_t = torch.linspace(beta_start, beta_end, timestep + 1)
        sqrt_beta_t = torch.sqrt(beta_t)
        
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def forward(self, x, c):
        B, *_ = x.shape
        timesteps = torch.randint(1, self.timestep + 1, (B,)).to(self.device)
        noise = torch.randn_like(x)
        
        x_t = self.sqrtab[timesteps, None, None, None] * x + \
            self.sqrtmab[timesteps, None, None, None] * noise
            
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)
        return self.loss(noise, self.model(x_t, c, timesteps / self.timestep, context_mask))
    
    def sample(self, 
               num_samples, 
               size, 
               device, 
               guide_weights=0.0, 
               mid_result_count: int=10):
        x_i = torch.randn(num_samples, *size).to(device)
        c_i = torch.arange(0, 10).to(device)
        c_i = c_i.repeat(int(num_samples / c_i.shape[0]))
        
        context_mask = torch.zeros_like(c_i).to(device)
        
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[num_samples:] = 1.
        
        x_i_store = []
        mid_interval = cfg.timestep // (mid_result_count - 1)
        
        for i in range(self.timestep, 0, -1):
            t_is = torch.tensor([i / self.timestep]).to(device)
            t_is = t_is.repeat(num_samples, 1, 1, 1)
            
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(num_samples, *size).to(device) if i > 1 else 0
            
            eps = self.model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:num_samples]
            eps2 = eps[num_samples:]
            eps = (1 + guide_weights) * eps1 - guide_weights * eps2
            x_i = x_i[:num_samples]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + \
                self.sqrt_beta_t[i] * z
            if i % mid_interval == 0 or i == self.timestep:
                x_i_store.append(x_i.detach().cpu().numpy())
        x_i_store = np.array(x_i_store)
        
        return x_i, x_i_store
    