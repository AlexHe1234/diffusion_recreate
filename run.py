from lib.network import make_network
from config import cfg
import torch
import cv2
import numpy as np
import os


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def run():
    device = choose_device()
    model = make_network(cfg, device).to(device)
    model.eval()
    if cfg.pretrained is False:
        raise ValueError('must specify pretrained model!')
    model.load_state_dict(torch.load(cfg.model_dir))
    if cfg.exp == 'mnist':
        s = [1, 28, 28]
    elif cfg.exp == 'cifar':
        s = [3, 32, 32]
    else:
        raise NotImplementedError
    with torch.no_grad():
        result, _ = model.sample(cfg.num_classes*cfg.render_num, 
                                                s, 
                                                device, 
                                                cfg.guided_weight)
    board = np.zeros((cfg.render_num*s[1], cfg.num_classes*s[2], s[0]))
    result = result.cpu().numpy().transpose([0, 2, 3, 1])
    for i in range(cfg.render_num):
        for j in range(cfg.num_classes):
            board[s[1]*i:s[1]*(i+1), s[2]*j:s[2]*(j+1)] = result[cfg.render_num * j + i]
    cv2.imwrite('./output.jpg', board*255)


if __name__ == '__main__':
    run()
