from yacs.config import CfgNode as CN
import os


cfg = CN()

cfg.exp_name = 'test2'

# global
cfg.exp = 'cifar'
cfg.pretrained = True
cfg.model_dir = './result/model_cifar-test2_24900.pth'
cfg.epochs = 1000
cfg.batch_size = 1024
cfg.num_classes = 10
cfg.lr = 1e-4
cfg.log_iter = 10
cfg.save_iter = 100
cfg.save_dir = 'result'
cfg.eval_epoch = 1
cfg.guided_weight = 0.0
cfg.fix_random = False
cfg.dist = False
cfg.render_num = 10

# params
cfg.timestep = 400
cfg.in_channels = 3
cfg.mid_channels = 128

# process config
if not os.path.exists(cfg.save_dir):
    os.mkdir(cfg.save_dir)
    
assert (cfg.pretrained == True) == (cfg.model_dir is not None), \
    'if use pretrain, then model_dir should be specified'

cfg.exp_name = cfg.exp + '-' + cfg.exp_name

if cfg.exp == 'mnist':
    cfg.in_channels = 1
elif cfg.exp == 'cifar':
    cfg.in_channels = 3
 
