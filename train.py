from config import cfg
from model import DDPM
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import os


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def train():
    device = choose_device()
    model = DDPM(1e-4, 1e-2, device).to(device)
    
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('./mnist', train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    iter = 0
    # train loop
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.param_groups[0]['lr'] = cfg.lr * (1. - epoch / cfg.epochs)
        
        loss_ema = None
        
        for x, c in dataloader:
            x = x.to(device)
            c = c.to(device)
            loss = model(x, c)
            optimizer.zero_grad()
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            optimizer.step()
            
            if iter % cfg.log_iter == 0:
                print(f'epoch {epoch} iter {iter} loss {loss.item()} loss_ema {loss_ema}')
                
            if iter % cfg.save_iter == 0 and iter != 0:
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, f'model_{iter}.pth'))
                print(f'model {iter} saved at {cfg.save_dir}')
            
            iter += 1
        

if __name__ == '__main__':
    train()
