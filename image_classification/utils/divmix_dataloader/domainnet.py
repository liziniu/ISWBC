from __future__ import annotations

import os

import numpy as np
from PIL import Image
from utils.config import DATA_PATHS
from utils.data_utils import find_classes, make_dataset_from_dir, IMG_EXTENSIONS
from utils.divmix_dataloader.dataset import ExDataset


class DomainNetDataset(ExDataset):
    all_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    def __init__(self, split='train', domain='MNIST', full_set=False):
        super(DomainNetDataset, self).__init__(split)
        self.full_set = full_set
        self.base_path = DATA_PATHS['DomainNet']
        self.domain = domain
        self.domain_id = self.all_domains.index(domain)
        if full_set:
            classes, class_to_idx = find_classes(f"{self.base_path}/{domain}")
            self.text_labels = classes
            self.paths, self.labels = make_dataset_from_dir(f"{self.base_path}/{domain}",
                                                            class_to_idx, IMG_EXTENSIONS)
            self.num_classes = len(class_to_idx)
        else:
            self.paths, self.text_labels = np.load('{}/{}_{}.pkl'.format(
                DATA_PATHS['DomainNetPathList'],
                domain, split), allow_pickle=True)

            class_to_idx = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4,
                            'tiger': 5, 'whale': 6, 'windmill': 7, 'wine_glass': 8, 'zebra': 9}

            self.labels = [class_to_idx[text] for text in self.text_labels]
            self.num_classes = len(class_to_idx)

        # self.transform = transform
        self.targets = self.labels
        self.classes = np.unique(self.labels)

    def __getitem__(self, idx):
        site, cls, fname = self.paths[idx].split('/')[-3:]
        img_path = os.path.join(self.base_path, site, cls, fname)

        target = self.labels[idx]
        img = Image.open(img_path)
        data_dict = {'image': img, 'target': target, 'domain': self.domain_id}
        return data_dict

    def get_all_targets(self, indices):
        return self.targets[indices]

    def __len__(self):
        return len(self.paths)
