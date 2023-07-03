import torch
from config.default import cfg
import os
import torch.distributed as dist
from lib.dataset import make_data_loader
from lib.network import make_network
from lib.trainer import make_trainer
from lib.optimizer import make_optimizer
from lib.evaluator import make_evaluator


if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    train_loader = make_data_loader()
    val_loader = make_data_loader()
    network = make_network()
    trainer = make_trainer()
    optimizer = make_optimizer()
    scheduler = make_scheduler()
    recorder = make_recorder()
    evaluator = make_evaluator()
    
    

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    if cfg.distributed:
        local_rank = int(os.environ['LOCAL_RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()
    train()
        
        
if __name__ == '__main__':
    main()