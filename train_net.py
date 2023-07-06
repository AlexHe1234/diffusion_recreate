from config import cfg
import torch
from torch.utils.data import DataLoader
from torch import optim
from lib.utils.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from lib.network import make_network
from lib.utils.dataset import make_dataset
from torch import distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler


if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def train():
    if cfg.dist:
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        synchronize()

    device = choose_device()
    
    model = make_network(cfg, device)
    if cfg.pretrained:
        model.load_state_dict(torch.load(cfg.model_dir))
        print('loaded')
    model = model.to(device)

    dataset = make_dataset(cfg)
    if not cfg.dist:
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    else:
        distributed_sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=4, sampler=distributed_sampler)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(f'./runs/{cfg.exp_name}')
    
    start_epoch = 0
    if cfg.pretrained:
        pass
    trainer = Trainer(model, optimizer, cfg, dataloader, writer, device, start_epoch)
    
    # train loop
    for epoch in range(cfg.epochs):
        if cfg.dist:
            distributed_sampler.set_epoch(epoch)
        trainer.train()
        if epoch % cfg.eval_epoch == 0:
            trainer.evaluate()


if __name__ == '__main__':
    train()
