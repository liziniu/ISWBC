import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from utils.config import DATA_PATHS

from .dataset import ExDataset


class Cifar10Dataset(CIFAR10, ExDataset):
    """Split Cifar100 into 10 domains with 10 classes per domain"""
    # all_domains = ['c0-19', 'c20-39', 'c40-59', 'c60-79', 'c80-99']
    # num_classes = 20  # may not be correct
    num_classes = 10

    def __init__(self, split='train', transform=None, download=True, dict_item=True):
        super().__init__(DATA_PATHS['Cifar10'], train=split == 'train', transform=transform, download=download)
        self.split = split
        self.domain = 'cifar10'
        self.domain_id = 0
        self.dict_item = dict_item

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.dict_item:
            data_dict = {'image': img, 'target': target, 'domain': self.domain_id}
            return data_dict
        else:
            return img, target

    def get_all_targets(self, indices):
        if not isinstance(self.targets, np.ndarray):
            targets = np.array(self.targets)
        else:
            targets = self.targets
        return targets[indices]
