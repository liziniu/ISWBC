import os
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import download_and_extract_archive
# from .core import NoisySemiSupDataset
from utils.config import DATA_PATHS
from utils.divmix_dataloader.dataset import ExDataset


class MNIST(ExDataset):
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, split, download=True):
        super(MNIST, self).__init__(split)
        self.root = DATA_PATHS["MNIST"]
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.split = split
        if self.split == 'test':
            data_file = self.test_file
        elif self.split == 'train':
            data_file = self.training_file
        else:
            raise ValueError(f"split: {self.split}")
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.transform = transforms.Grayscale(num_output_channels=3)

        # root_dir = os.path.join(self.root, self.base_folder)
        # self.r = r  # noise ratio
        # self.transform = transform
        # self.mode = mode
        # self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
        #                    8: 8}  # class transition for asymmetric noise

        # super(MNIST, self).__init__(data, targets, mode, transform, **kwargs)

    def __len__(self):
        return len(self.data)

    #     if self.mode != 'test':
    #         return len(self.train_data)
    #     else:
    #         return len(self.test_data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    # def extra_repr(self) -> str:
    #     return "Mode: {}".format(self.mode)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = array2image(img)
        img = self.transform(img)
        return img, target

    def get_all_targets(self, indices):
        return self.targets[indices]


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
