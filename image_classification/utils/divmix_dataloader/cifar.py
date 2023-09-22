import os

from PIL import Image
from torchvision.datasets import CIFAR10
from utils.config import DATA_PATHS


class CifarDataset(CIFAR10):
    all_domains = ['Cifar10', 'Cifar100']
    num_classes = 10  # may not be correct

    def __init__(self, domain='cifar10', split='train', transform=None, download=False):
        assert domain in self.all_domains, f"Invalid domain: {domain}"
        data_path = os.path.join(DATA_PATHS[domain])
        super().__init__(data_path, train=split == 'train', transform=transform, download=download)
        self.split = split
        self.domain = domain
        self.domain_id = self.all_domains.index(domain)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        data_dict = {'image': img, 'target': target, 'domain': self.domain_id}
        return data_dict

    def get_all_targets(self, indices):
        return self.targets[indices]
