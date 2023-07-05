from yacs.config import CfgNode as CN
import os


cfg = CN()

cfg.exp_name = 'test'

# global
cfg.pretrained = False
cfg.model_dir = None
cfg.epochs = 20
cfg.batch_size = 256
cfg.num_classes = 10
cfg.lr = 1e-4
cfg.log_iter = 10
cfg.save_iter = 10
cfg.save_dir = 'result'


# params
cfg.timestep = 400
cfg.in_channels = 1
cfg.mid_channels = 128

# process config
if not os.path.exists(cfg.save_dir):
    os.mkdir(cfg.save_dir)
    
assert (cfg.pretrained == True) == (cfg.model_dir is not None), \
    'if use pretrain, then model_dir should be specified'
 
