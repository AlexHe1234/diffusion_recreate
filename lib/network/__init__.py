def make_network(cfg, device):
    if cfg.exp == 'mnist':
        from .mnist import DDPM
        return DDPM(1e-4, 1e-2, device)
    elif cfg.exp == 'cifar':
        from .cifar import DDPM
        return DDPM(1e-4, 1e-2, device)
    else:
        raise NotImplementedError
