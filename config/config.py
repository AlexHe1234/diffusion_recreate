from yacs.config import CfgNode as CN
import yaml
import argparse
import os


cfg = CN()

cfg.distributed = False
cfg.fix_random = False
cfg.exp_name = 'default'

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='default.yaml', help='config yaml file name')

args = parser.parse_args()

# open config yaml file
config_file = __file__[:-9] + args.config
if '.yaml' not in config_file:
    config_file += '.yaml'
    
# print(config_file)
if not os.path.exists(config_file):
    raise Exception("config yaml file doesn't exist")

yaml_file = open(config_file, 'r')
yaml_open = yaml.safe_load(yaml_file)

# replace default value
default_keys = list(cfg.keys())
for k in yaml_open.keys():
    if k in default_keys:
        cfg.__setattr__(k, yaml_open[k])
        
