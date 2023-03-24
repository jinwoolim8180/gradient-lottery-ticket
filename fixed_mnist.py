from torchvision import transforms, datasets
import math
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class FixedMNIST(Dataset):
    def __init__(self, dir, train=True):
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )
    
        # basic MNIST dataset
        self.dataset = datasets.MNIST(dir, train=train, download=True, transform=apply_transform)
        self.one_indices = (self.dataset.targets == 1).nonzero(as_tuple=True)
        self.five_indices = (self.dataset.targets == 5).nonzero(as_tuple=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset.data[idx], self.dataset.targets[idx]
        if target == 0:
            img = self.dataset.data[random.randrange(len(self.five_indices))]
            img += self.dataset.data[random.randrange(len(self.one_indices))]
        return img.type('torch.FloatTensor').unsqueeze(0), target