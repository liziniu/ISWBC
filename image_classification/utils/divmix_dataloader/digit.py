from __future__ import annotations

import bisect
import os
from collections import defaultdict
from typing import Dict
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms
from utils.config import DATA_PATHS
from utils.divmix_dataloader.dataset import ExDataset


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def get_constant_transforms(domain):
    """Only non-variable transforms."""
    trns = {
        'MNIST-F': [
            transforms.Grayscale(num_output_channels=3),
            # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ],
        'MNIST': [
            transforms.Grayscale(num_output_channels=3),
        ],
        'SVHN': [
            transforms.Resize([28, 28]),
        ],
        'USPS': [
            transforms.Resize([28, 28]),
            transforms.Grayscale(num_output_channels=3),
        ],
        'SynthDigits': [
            transforms.Resize([28, 28]),
        ],
        'MNIST_M': [
            transforms.Lambda(lambda x: x)  # identity
        ],
    }
    return transforms.Compose(trns[domain])


class DigitsDataset(ExDataset):
    all_domains = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M']
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, split='train', domain='MNIST', percent=1., max_n_test=5_000):
        super(DigitsDataset, self).__init__(split)
        # self.root = root_dir
        data_path = os.path.join(DATA_PATHS["Digits"], domain)
        if split == 'test':
            self.data, self.targets = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
            if max_n_test > 0:
                n_test = max_n_test  # int(len(self.targets) * test_percent)
                self.data, self.targets = self.data[:n_test], self.targets[:n_test]
                assert len(np.unique(self.targets)) == 10, f"Not enough classes: {np.unique(self.targets)}"
                # print(f"@### domain {domain} n_test: {n_test}")
        elif split == 'train':
            if percent >= 0.1:
                for part in range(int(percent * 10)):
                    if part == 0:
                        self.data, self.targets = np.load(
                            os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)),
                            allow_pickle=True)
                    else:
                        data, targets = np.load(
                            os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)),
                            allow_pickle=True)
                        self.data = np.concatenate([self.data, data], axis=0)
                        self.targets = np.concatenate([self.targets, targets], axis=0)
            else:
                self.data, self.targets = np.load(
                    os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                data_len = int(self.data.shape[0] * percent * 10)
                self.data = self.data[:data_len]
                self.targets = self.targets[:data_len]
        else:
            raise NotImplementedError(f"split: {split}")
        self.domain = domain
        self.domain_id = self.all_domains.index(domain)
        self.labels = self.targets.astype(np.long).squeeze()
        self.channels = 3 if domain in ['SVHN', 'SynthDigits', 'MNIST_M'] else 1
        self.transform = get_constant_transforms(domain)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = array2image(img)
        img = self.transform(img)
        data_dict = {'image': img, 'target': target, 'domain': self.domain_id}
        return data_dict

    def get_all_targets(self, indices):
        return self.targets[indices]

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}


class CatDigitsDataset(ExDataset):
    def __init__(self, datasets: List[DigitsDataset]):
        super(CatDigitsDataset, self).__init__(split=datasets[0].split)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"

        all_data = defaultdict(list)
        for ds in self.datasets:
            # all_data['data'].append(ds.data)  # need to run transform
            all_data['targets'].append(ds.targets)
        self.check_consistency(self.datasets)

        self.split = self.datasets[0].split
        # self.data = all_data['data']
        self.targets = np.concatenate(all_data['targets'], axis=0)
        self.domain = '+'.join([ds.domain for ds in self.datasets])
        self.cumulative_sizes = self.cumsum(self.datasets)

    datasets: List[DigitsDataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_all_targets(self, indices):
        return self.targets[indices]

    @staticmethod
    def check_consistency(datasets: List[DigitsDataset]):
        if len(datasets) == 1:
            return
        else:
            for ds in datasets[1:]:
                assert ds.split == datasets[0].split
            u_domains = np.unique([ds.domain for ds in datasets])
            assert len(u_domains) == len(datasets), f"Found duplicated domains: {u_domains}"


def array2image(image):
    if len(image.shape) == 2:
        # image = Image.fromarray(image, mode='L')
        # FIXME ad-hoc to make a 3-channel data (otherwise, we canot do some aug)
        image = Image.fromarray(np.stack([image for _ in range(3)], axis=-1), mode='RGB')
    else:
        image = Image.fromarray(image, mode='RGB')
    # else:
    #     raise ValueError("{} channel is not allowed.".format(self.channels))
    return image
