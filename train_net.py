from config import cfg
from model import DDPM
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter


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
    dataset = MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(f'./runs/{cfg.exp_name}')
    
    start_epoch = 0
    if cfg.pretrained:
        pass
    trainer = Trainer(model, optimizer, cfg, dataloader, writer, device, start_epoch)
    
    # train loop
    for epoch in range(cfg.epochs):
        trainer.train()
        if epoch % cfg.eval_epoch == 0:
            trainer.evaluate()


if __name__ == '__main__':
    train()
