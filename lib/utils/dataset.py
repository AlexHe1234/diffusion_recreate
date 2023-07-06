from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision import transforms


def make_dataset(cfg):
    if cfg.exp == 'mnist':
        return MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())
    elif cfg.exp == 'cifar':
        return CIFAR10('./data/cifar10', train=True, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError
    