import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100
from utils.config import DATA_PATHS

from .dataset import ExDataset


class Cifar100C20Dataset(CIFAR100, ExDataset):
    """Split Cifar100 into 10 domains with 10 classes per domain"""
    all_domains = ['c0-19', 'c20-39', 'c40-59', 'c60-79', 'c80-99']
    num_classes = 20  # may not be correct

    def __init__(self, domain='c0-19', split='train', transform=None, download=True, dict_item=True):
        assert domain in self.all_domains, f"Invalid domain: {domain}"
        super().__init__(DATA_PATHS['Cifar100'], train=split == 'train', transform=transform, download=download)
        self.split = split
        self.domain = domain
        self.domain_id = self.all_domains.index(domain)

        start, end = domain[1:].split('-')
        sel_classes = np.arange(int(start), int(end) + 1)

        self.classes = [self.classes[c] for c in sel_classes]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        select = np.isin(self.targets, sel_classes)
        list_sel_classes = sel_classes.tolist()
        self.targets = [list_sel_classes.index(self.targets[i]) for i in np.where(select)[0]]
        self.data = self.data[select]
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
        return self.targets[indices]
