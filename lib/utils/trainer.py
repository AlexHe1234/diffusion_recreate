import os
import torch
import torchvision


class Trainer(object):
    def __init__(self, 
                 model, 
                 optimizer, 
                 cfg, 
                 dataloader, 
                 writer,
                 device, 
                 start_epoch=0, 
                 end_epoch=500):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.epoch = start_epoch
        self.dataloader = dataloader
        self.device = device
        self.iter = 0
        self.end_epoch = end_epoch
        self.writer = writer
    
    def train(self):
        self.model.train()
        self.optimizer.param_groups[0]['lr'] = self.cfg.lr * (1. - self.epoch / self.cfg.epochs)
        
        loss_ema = None
        
        for x, c in self.dataloader:
            x = x.to(self.device)
            c = c.to(self.device)
            loss = self.model(x, c)
            self.optimizer.zero_grad()
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            self.optimizer.step()

            if self.cfg.local_rank == 0:
                if self.iter % self.cfg.log_iter == 0:
                    print(f'epoch {self.epoch} iter {self.iter} loss {loss.item()} loss_ema {loss_ema}')
                    
                if (self.iter % self.cfg.save_iter == 0 and self.iter != 0) or self.epoch == self.end_epoch:
                    torch.save(self.model.state_dict(), os.path.join(self.cfg.save_dir, f'model_{self.cfg.exp_name}_{self.iter}.pth'))
                    print(f'model {self.iter} saved at {self.cfg.save_dir}')
            
            self.iter += 1
            
        self.epoch += 1
    
    @torch.no_grad()
    def evaluate(self, group=1):
        self.model.eval()
        if self.cfg.exp == 'mnist':
            s = [1, 28, 28]
        elif self.cfg.exp == 'cifar':
            s = [3, 32, 32]
        else:
            raise NotImplementedError
        result, mid_results = self.model.sample(self.cfg.num_classes*group, 
                                               s, 
                                               self.device, 
                                               self.cfg.guided_weight)
        result_grid = torchvision.utils.make_grid(result, normalize=True)
        self.writer.add_image('evaluation', result_grid, global_step=self.iter)
    