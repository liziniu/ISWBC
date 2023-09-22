from functools import partial
import os
import numpy as np
import PIL
import pandas
import pandas as pd
import torch
from argparse import ArgumentParser
import hashlib
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, \
    extract_archive, list_files, download_file_from_google_drive, download_and_extract_archive
import torchvision.transforms as trns
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import T_co
from PIL import Image
from collections import defaultdict

from utils.divmix_dataloader.dataset import ExDataset
from .config import data_root
from .url_defense_decoder import URLDefenseDecoder


class IndexedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset[T_co]):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx], idx

    def __len__(self):
        return len(self.dataset)


def check_integrity_sha256(fpath: str, sha256: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if sha256 is None:
        return True
    if not check_sha256(fpath, sha256):
        print(f" Fail to match SHA256 of {fpath}")
        return False
    return True


def calculate_sha256(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.sha256()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_sha256(fpath: str, sha256: str, **kwargs: Any) -> bool:
    print(f" Verifying SHA256 of {fpath}")
    cal_val = calculate_sha256(fpath, **kwargs)
    return sha256 == cal_val


class IJB(VisionDataset):
    """The IJB dataset. The dataset is also named IJB-C which contains -A, -B variants.
    Request dataset at https://www.nist.gov/programs-projects/face-challenges
    """
    base_folder = "IJB"
    # file links have to be updated by sending request to NIST. Below links are invalid
    # after Nov. 5th.
    content_list_file = "7liQtxINaU7aPnTjm61Osv7F7vw_Rg4QjKdDzoHBcLQ*3D"
    file_list = [
        # token,  SHA-256, file name
        (
            "luHbS6_ovmcelrtaEc-6CqIqq6iAAINzVB2vn8KyAR8*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxdWAOTSSA$",
            "05b8800797e977a45012df2a9f6e1aa2a6ac0992b4481872d2c628dc91f9feed"),
        (
            "WR-hRWIeA51X8FrEFCm954LuLU1ctf6gd5qwzj2ziZY*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxfFVqtdVQ$",
            "19c116292580fca69f992301d956e84c5a3358deea98c0d0f1341163508f3061"),
        (
            "5AfIyiRDA1XB8rUc5fZQOOiuwLcMYenM6x3prNQfBdI*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxcOMVzW5Q$",
            "2272c308998afc6c2f6aa283e60ea0e16f826be0aefaf453033d18cd8aadc801"),
        (
            "GUHK9_NyjIppdNRqTdJ3k_qIITL5Z6drqtMTcRUIbsY*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxfh2CUzcA$",
            "11c47f7f8e95d8b98929f1709af92dec2fe3c638f26cd5bfcd7e7501470b4923"),
        (
            "soHJLQYFsKo-BbUTt8jG2tjxRUl2gxiDeww0oENavi8*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxdrImk7sw$",
            "22d578ee0a806a14ab6ddf28ff30e23ed9b5aaae0262258b6e02b599496f0209"),
        (
            "vQ2TLS2jreO_s2Y8xVAgbpKKF3NQK7LmuFU6g2-BazI*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxfsuxtvMA$",
            "efbe3ae24933f94fac53b0fe5ee23b4c300279ff90ba74264cc3569b2535574e"),
        (
            "wLeRMm9BRgcgr8g3X4OiOCRgiyCIRGTalatC-1y8yp8*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxeMu-pyPA$",
            "fdb4f85c545810b025db8212854831b483659fd977a217bb038b74960b8f388e"),
        (
            "HAygWBpouJOmAi2CyFUbtA6wD3KDPHdqEFHEReXqcys*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxdyNL8ioQ$",
            "a02d84749780670fba8339999d2da7f9627e90dce7a6d939af626c7263dcac33"),
        (
            "68t0ONqv3Udc_ikl5gIpEYsFsJN641EkMS-ZVqdpa5o*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxeNa4M5ig$",
            "8d64df599d5aa9d173407731dd62d1151451f64ee07083a3f3ae7b07454658f1"),
        (
            "MWTqidescu2frV_DK5jf0xhN35C3A3HOS_EsGKvqK5w*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxemd7gn8Q$",
            "28f0d65c36345ab4a55cf7eb0d5ef9041f71c3db9da8f541d44526db249e19a7"),
        (
            "sNYjvJVp79ZmZY6x9oeG7N0kN9Dn-BJqSCjx0feELJA*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxd9UD5_yA$",
            "501e3268bc6c5cd0f10f1abb8a6eab49b1487e18f40c40703ebc0a49d258664b"),
        (
            "phoOwBR1Us5l3wm4Cycfv6ixVdyjM8FSnemZLVX8Ayo*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxf73_FY0w$",
            "ab5f37f03ad56002d072973597fae5071211450519ea212501e944387e32ff87"),
        (
            "yYn0D2WJ6ATwdv5iAMM_KVb7DEyI5-ektCtJruzBJAw*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxdQsg0Rug$",
            "08717a1358a96d99483637e1b27a4ca87d345a585ecb3e25c225b4c3d78c78b7"),
        (
            "b7Vs8EpuI2P2diZtB0F6DO96doMGytjU2pALuHDClyM*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxcB0umUPw$",
            "2cffb27e44e1a9632c3a16a96929420f73d5ad946b9c0cd2c51eaa6d601ce9ac"),
        (
            "t5VimWQBtRz31jwz17y6eiNb0QCmG5HX6foF3toMFIo*3D__;JQ!!HXCxUKc!jhH2V91da3-pngp8y3smIYODU4adGp1WxyjMsFAMF4JSWdtxzimLpc0TCxeGoJrT-A$",
            "a09eecf5a685e53576b0a4a8466d133c0f1f0b733d2fd8397b009ad755ab0ce3"),
    ]
    cat_file_sha256 = "dcf3c16f82693155c68712e60bda9cefd39b6d633380964df50e0f785648f721"

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            strict_file_check: bool = False,
    ) -> None:
        super(IJB, self).__init__(root, transform=transform,
                                  target_transform=target_transform)
        self.split = split
        self._strict_file_check = strict_file_check
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # split_map = {
        #     "train": 0,
        #     "valid": 1,
        #     "test": 2,
        #     "all": None,
        # }
        # split_ = split_map[verify_str_arg(split.lower(), "split",
        #                                   ("train", "valid", "test", "all"))]

        # fn = partial(os.path.join, self.root, self.base_folder)
        # splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        # identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        # bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        # landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        # attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        #
        # mask = slice(None) if split_ is None else (splits[1] == split_)
        #
        # self.filename = splits[mask].index.values
        # self.identity = torch.as_tensor(identity[mask].values)
        # self.bbox = torch.as_tensor(bbox[mask].values)
        # self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        # self.attr = torch.as_tensor(attr[mask].values)
        # self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        # self.attr_names = list(attr.columns)

    def complete_url(self, token):
        def_url = "https://urldefense.com/v3/__https://nigos.nist.gov/datasets/ijbc/download?token=" + token
        return URLDefenseDecoder().decode(def_url)

    def complete_filename(self, file_idx):
        return f"IJB.tar.gz.{file_idx:02d}"

    def _check_integrity(self) -> bool:
        for file_idx, (_, sha256) in enumerate(self.file_list):
            filename = self.complete_filename(file_idx)
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            if self._strict_file_check and not check_integrity_sha256(fpath, sha256):
                print(f"Invalid SHA256 for {fpath}")
                return False

        # TODO Should check a hash of the images
        # return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))
        return True

    def download(self) -> None:
        download_root = os.path.expanduser(self.root)
        download_root = os.path.join(download_root, self.base_folder)
        fpath_list = []
        for file_idx, (token, sha256) in enumerate(self.file_list):
            filename = self.complete_filename(file_idx)
            fpath = os.path.join(download_root, filename)
            fpath_list.append(fpath)
        if self._check_integrity():
            print('Files already downloaded and verified')
            # return
        else:
            for file_idx, (token, sha256) in enumerate(self.file_list):
                filename = self.complete_filename(file_idx)
                fpath = os.path.join(download_root, filename)
                url = self.complete_url(token)
                download_url(url, download_root, filename)

                if not check_integrity_sha256(fpath, sha256):
                    raise RuntimeError("File not found or corrupted.")

        cat_fpath = os.path.join(download_root, "IJB.tar.gz")
        if self._strict_file_check and not check_integrity_sha256(cat_fpath, self.cat_file_sha256):
            cmd = "cat " + " ".join(fpath_list) + " > " + cat_fpath
            print(cmd)
            os.system(cmd)

            print(f"Extract files into {download_root}")
            extract_archive(cat_fpath, download_root, remove_finished=True)
        else:
            return
            # raise RuntimeError(f"Corrupted cat file: {cat_fpath}")


class FaceDataset(VisionDataset):
    def stat_attr(self):
        raise NotImplementedError()


class UTKFace(FaceDataset, ExDataset):
    """Have to download files in advance.
    UTKFace dataset: https://susanqq.github.io/UTKFace/
    Cite:
    @inproceedings{zhifei2017cvpr,
      title={Age Progression/Regression by Conditional Adversarial Autoencoder},
      author={Zhang, Zhifei, Song, Yang, and Qi, Hairong},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2017},
      organization={IEEE}
    }

    """
    base_folder = 'UTKFace'
    available_attr_to_i = {'age': 0, 'gender': 1, 'race': 2}

    def __init__(self,
                 root: str,
                 target_type: str = "gender",
                 return_attr: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 categorical_age=True,
                 dict_item=True):
        super(UTKFace, self).__init__(os.path.join(root, self.base_folder, self.base_folder), transform=transform,
                                      target_transform=target_transform)

        # NOTE the order corresponds to the order of attr in a filename.
        self.attr_names = ['age', 'gender', 'race']
        if target_type == 'age':
            raise NotImplementedError("Not implement for age yet.")
        self.target_type = target_type
        self.attr_classes = {
            'race': ('White', 'Black', 'Asian', 'Indian', 'Others'),
            'gender': ('Male', 'Female'),
            'age': ('0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '>=70')
            if categorical_age else list(range(117)),  # categorical based on FairFace
        }
        assert self.attr_classes['age'][-1] == '>=70'  # otherwise the last in the below list will be wrong.
        self._least_ages = np.array([int(ag.split('-')[1]) if '-' in ag else float('inf') for ag in self.attr_classes['age']])
        self.attr_classes_to_idx = {
            a: {c: ic for ic, c in enumerate(a_c)}
            for a, a_c in self.attr_classes.items()
        }

        self.filename = list_files(self.root, 'jpg', prefix=False)
        self.attr, self.filename = self._extract_attr_from_filenames(self.filename)
        self.attr, self.attr_names = self._make_binary_attr(self.attr, self.attr_names)
        self.return_attr = return_attr

        assert target_type in self.attr_names
        self.classes = self.attr_classes[self.target_type]

        self.dict_item = dict_item
        self.domain = 'UTKFace'
        self.domain_id = 0
        self.split = 'train'

    def _extract_attr_from_filenames(self, filenames):
        attr = np.zeros((len(filenames), len(self.attr_names)), dtype=int)
        valid_idxs, valid_filenames = [], []
        for idx, fname in enumerate(filenames):
            for ai, a in enumerate(self.attr_names):
                s = fname.split('_')
                if len(s) < 4:
                    continue
                if a == 'age':
                    age = int(s[ai])
                    age_cat = np.argmax(age <= self._least_ages)
                    attr[idx, ai] = age_cat
                else:
                    attr[idx, ai] = int(s[ai])
                valid_filenames.append(fname)
                valid_idxs.append(idx)
        attr = attr[valid_idxs, :]
        return attr, valid_filenames

    def _make_binary_attr(self, attr, attr_names):
        attr_names += ['young']
        attr = np.concatenate([attr, np.isin(attr[:, attr_names.index('age')], [0, 1, 2, 3]).astype('int').reshape((-1, 1))], axis=1)
        self.attr_classes['young'] = ('old', 'young')
        self.attr_classes_to_idx['young'] = {'old': 0, 'young': 1}

        sup_a = 'race'
        for a in self.attr_classes[sup_a]:
            attr_names += [a]
            attr = np.concatenate([
                attr, (attr[:, attr_names.index(sup_a)] == self.attr_classes_to_idx[sup_a][a]).astype('int').reshape((-1, 1))], axis=1)
            self.attr_classes[a] = ('neg', 'pos')
            self.attr_classes_to_idx[a] = {'neg': 0, 'pos': 1}

        sup_a = 'age'
        for a in self.attr_classes[sup_a]:
            attr_names += [sup_a + a]
            attr = np.concatenate([
                attr, (attr[:, attr_names.index(sup_a)] == self.attr_classes_to_idx[sup_a][a]).astype('int').reshape((-1, 1))], axis=1)
            self.attr_classes[sup_a + a] = ('neg', 'pos')
            self.attr_classes_to_idx[sup_a + a] = {'neg': 0, 'pos': 1}
        return attr, attr_names

    def __getitem__(self, idx):
        filename = self.filename[idx]  # type: str
        fpath = os.path.join(self.root, filename)
        img = PIL.Image.open(fpath)
        img = img.convert('RGB')

        # attrs = {a: v for a, v in zip(self.attr_names, self.attr[idx, :])}
        attrs = self.attr[idx, :]
        target = attrs[self.attr_names.index(self.target_type)]

        # transform
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.dict_item:
            if self.return_attr:
                d = {'image': img, 'attr': attrs, 'domain': 0}
            else:
                d = {'image': img, 'target': target, 'domain': 0}
            return d
        else:
            if self.return_attr:
                return img, attrs
            else:
                return img, target

    def __len__(self) -> int:
        return len(self.filename)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    # def stat_attr(self):
    #     df_dict = {"attr_name": [], "-": [], "+": [], "pos_rate": []}
    #     for ai, a in enumerate(self.attr_names):
    #         a_list = self.attr[:, ai]
    #         df_dict['attr_name'].append(a)
    #         df_dict['+'].append(np.sum(a_list > 0))
    #         df_dict['-'].append(np.sum(a_list < 1))
    #         df_dict['pos_rate'].append(np.sum(a_list > 0)/len(a_list))
    #     print(pandas.DataFrame(df_dict).set_index('attr_name'))

    def stat_attr(self):
        for ai, a in enumerate(self.attr_names):
            a_list = self.attr[:, ai]
            info_c = ' '.join([f"{c} ({ci}): {np.sum(a_list == ci)} ({np.mean(a_list == ci) * 100:.1f}%)" for c, ci in
                               self.attr_classes_to_idx[a].items()])
            print(f"{a:12s}\t{info_c}")
        A1 = 'gender'
        A2 = 'race'
        a = self.attr[:, [self.attr_names.index(A1), self.attr_names.index(A2)]]
        # paired_classes = {(c1, c2): (ci1, ci2) for c2, ci2 in self.attr_classes_to_idx[A2].items()
        #                   for c1, ci1 in self.attr_classes_to_idx[A1].items()}
        info = ''
        for c1, ci1 in self.attr_classes_to_idx[A1].items():
            info += '\n'
            for c2, ci2 in self.attr_classes_to_idx[A2].items():
                mask = (a[:, 0] == ci1) & (a[:, 1] == ci2)
                info += f" {c1:>6s},{c2:>6s} ({ci1},{ci2}): {np.sum(mask)} ({np.mean(mask) * 100:.1f}%)"
        # info_c = ' '.join([f"{c}: {np.sum(a_list == ci)} ({np.mean(a_list == ci)*100:.1f}%)" for c, ci in self.attr_classes_to_idx[a].items()])
        print(f"({A1}, {A2})\t{info}")

    def get_all_targets(self, indices):
        return self.targets[indices]

    @property
    def targets(self):
        return self.attr[:, self.attr_names.index(self.target_type)]  # .numpy()


class FairFace(FaceDataset):
    """Here we use 0.25 padding dataset.
    Data: https://github.com/dchen236/FairFace

    Ref: Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), 1547–1557. https://doi.org/10.1109/WACV48630.2021.00159
    """
    base_folder = 'FairFace'
    file_list = [
        # File ID                             # filename
        # ("1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL", "fairface-img-margin125-trainval.zip"),
        ("1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86", "fairface-img-margin025-trainval.zip"),
        ("1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH", "fairface_label_train.csv"),
        ("1wOdja-ezstMEp81tX1a-EYkFebev4h7D", "fairface_label_val.csv"),
    ]

    def __init__(self,
                 root: str,
                 split='train',
                 target_type: str = "gender",  # age, gender, race, service_test
                 return_attr: bool = True,
                 download=False,
                 use_utk_race=True,  # simplified race partition.
                 # margin=0.25,  # select one preprocessing set. 0.25 (default) or 1.25
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super(FairFace, self).__init__(os.path.join(root, self.base_folder), transform=transform, target_transform=target_transform)
        assert split in ('train', 'val'), f'Invalid split: {split}'
        self.split = split  # train or val
        self.target_type = target_type
        self.return_attr = return_attr
        self.use_utk_race = use_utk_race
        # self._margin = 0.25
        # if self._margin == 0.25:  NOTE should not choose margin as we unzip train and val in-place which means two sets cannot exist meantime.
        self._data_zip_file = "fairface-img-margin025-trainval.zip"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.filename = list_files(os.path.join(self.root, self.split), '.jpg', prefix=False)
        self.filename.sort(key=lambda s: int(s[:-len('.jpg')]))
        fn = partial(os.path.join, self.root)
        str_attr_df = pandas.read_csv(fn(f"fairface_label_{self.split}.csv"), header=0, index_col=0)
        self.attr_names = str_attr_df.columns.values.tolist()

        self.attr_classes = {
            'gender': ('Male', 'Female'),
            'race': ('White', 'Black', 'East Asian', 'Indian', 'Middle Eastern',
                     'Latino_Hispanic', 'Southeast Asian'),
            'age': ('0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '>=70'),
            'service_test': [True, False],
        }
        self.to_utk_race = {
            'White': 'White', 'Black': 'Black', 'Indian': 'Indian',
            'East Asian': 'Asian', 'Southeast Asian': 'Asian',
            'Middle Eastern': 'Others', 'Latino_Hispanic': 'Others',
        }
        if self.use_utk_race:
            self.attr_classes['race'] = ('White', 'Black', 'Asian', 'Indian', 'Others')
        self.classes = self.attr_classes[self.target_type]

        self.attr_classes_to_idx = {
            a: {c: ic for ic, c in enumerate(a_c)}
            for a, a_c in self.attr_classes.items()
        }
        attr_fp = fn(f'fairface_attr_{self.split}.csv')
        if os.path.exists(attr_fp):
            print(f"Load attr from {attr_fp}")
            # TODO verify head
            attr_df = pd.read_csv(attr_fp)
            assert np.all(attr_df.columns.values[1:] == self.attr_names), f"Mismatched attr names: {attr_df.columns}"
            self.attr = attr_df.values[:, 1:]  # remove index column
            verify_dict = {'train': np.array([6, 0, 2, 0]), 'val': np.array([1, 0, 2, 1])}
            assert np.all(
                self.attr[0] == verify_dict[self.split]), f"Verification failed: {self.attr[0]} Expected: {verify_dict[self.split]}"
            assert len(self.filename) == len(self.attr), f"Mismatched size: {len(self.filename)} vs {len(self.attr)}"
        else:
            self.attr = self._extract_attr(self.filename, str_attr_df)
            pd.DataFrame(self.attr, columns=self.attr_names).to_csv(attr_fp)
            # np.savetxt(attr_fp, self.attr, header=','.join(self.attr_names), delimiter=',', fmt='%2d')
            print(f"Cache attr into {attr_fp}")
            # TODO cache
        self.attr, self.attr_names = self._make_binary_attr(self.attr, self.attr_names)
        assert self.target_type in self.attr_names, f"target type {self.target_type} not in {self.attr_names}"
        self.attr = torch.tensor(self.attr)

    def _extract_attr(self, filenames, str_attr_df):
        attr = np.zeros((len(filenames), len(self.attr_names)), dtype=int)
        print(f"Transform attribute str into numbers")
        for idx, fname in enumerate(tqdm(filenames)):
            fid = os.path.join(self.split, fname)
            fs_attr = str_attr_df.loc[fid]
            for ai, a in enumerate(self.attr_names):
                s_attr = fs_attr[a]
                if self.use_utk_race and a == 'race':
                    s_attr = self.to_utk_race[s_attr]
                if s_attr == 'more than 70':
                    s_attr = '>=70'
                attr[idx, ai] = self.attr_classes_to_idx[a][s_attr]
        return attr

    def _make_binary_attr(self, attr, attr_names):
        attr_names += ['young']
        attr = np.concatenate([attr, np.isin(attr[:, attr_names.index('age')], [0, 1, 2, 3]).astype('int').reshape((-1, 1))], axis=1)
        self.attr_classes['young'] = ('old', 'young')
        self.attr_classes_to_idx['young'] = {'old': 0, 'young': 1}

        sup_a = 'race'
        for a in self.attr_classes[sup_a]:
            attr_names += [a]
            attr = np.concatenate([
                attr, (attr[:, attr_names.index(sup_a)] == self.attr_classes_to_idx[sup_a][a]).astype('int').reshape((-1, 1))], axis=1)
            self.attr_classes[a] = ('neg', 'pos')
            self.attr_classes_to_idx[a] = {'neg': 0, 'pos': 1}

        sup_a = 'age'
        for a in self.attr_classes[sup_a]:
            attr_names += [sup_a + a]
            attr = np.concatenate([
                attr, (attr[:, attr_names.index(sup_a)] == self.attr_classes_to_idx[sup_a][a]).astype('int').reshape((-1, 1))], axis=1)
            self.attr_classes[sup_a + a] = ('neg', 'pos')
            self.attr_classes_to_idx[sup_a + a] = {'neg': 0, 'pos': 1}
        return attr, attr_names

    def __getitem__(self, idx: int):
        img = PIL.Image.open(os.path.join(self.root, self.split, self.filename[idx]))

        # attrs = {a: v for a, v in zip(self.attr_names, self.attr[idx, :])}
        attrs = self.attr[idx, :]
        target = attrs[self.attr_names.index(self.target_type)]

        # transform
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_attr:
            return img, attrs
        else:
            return img, target

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, filename) in self.file_list:
            download_file_from_google_drive(file_id, self.root, filename)

        print(f"Unzip file: {os.path.join(self.root, self._data_zip_file)}")
        with zipfile.ZipFile(os.path.join(self.root, self._data_zip_file), "r") as f:
            f.extractall(self.root)

    def _check_integrity(self) -> bool:
        for (_, filename) in self.file_list:
            fpath = os.path.join(self.root, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, "train"))

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    def stat_attr(self):
        attr = self.attr if not isinstance(self.attr, torch.Tensor) else self.attr.numpy()
        for ai, a in enumerate(self.attr_names):
            a_list = attr[:, ai]
            info_c = ' '.join(
                [f"{c}: {np.sum(a_list == ci)} ({np.mean(a_list == ci) * 100:.1f}%)" for c, ci in self.attr_classes_to_idx[a].items()])
            print(f"{a:12s}\t{info_c}")

        # df_dict = {"attr_name": [], "-": [], "+": [], "pos_rate": []}
        # for ai, a in enumerate(self.attr_names):
        #     a_list = self.attr[:, ai]
        #     df_dict['attr_name'].append(a)
        #     df_dict['+'].append(np.sum(a_list > 0))
        #     df_dict['-'].append(np.sum(a_list < 1))
        #     df_dict['pos_rate'].append(np.sum(a_list > 0)/len(a_list))
        # print(pandas.DataFrame(df_dict).set_index('attr_name'))

    def stat_one_attr(self, attr_name, indexes=None):
        ai = self.attr_names.index(attr_name)
        a_list = self.attr[:, ai].numpy()
        if indexes is not None:
            a_list = a_list[indexes]
        n_pos = np.sum(a_list > 0)
        n_neg = np.sum(a_list < 1)
        pos_rate = n_pos * 1. / len(a_list)
        return pos_rate

    def bias_attr(self, attr_name, n_sample, pos_rate):
        """Resample the attributes."""
        ai = self.attr_names.index(attr_name)
        a_list = self.attr[:, ai].numpy()
        n_pos = np.sum(a_list > 0)
        n_neg = np.sum(a_list < 1)

        assert n_sample <= len(self), "Not enough samples."
        sel_n_pos = int(n_sample * pos_rate)
        assert sel_n_pos < n_pos, f"Not enough positive samples. Have {n_pos} pos samples," \
                                  f" but wanted {sel_n_pos}."
        pos_idxs = np.random.choice(n_pos, sel_n_pos, replace=False)
        pos_idxs = np.nonzero(a_list > 0)[0][pos_idxs].tolist()
        sel_n_neg = int(n_sample * (1 - pos_rate))
        assert sel_n_neg < n_neg, f"Not enough positive samples. Have {n_neg} pos samples," \
                                  f" but wanted {sel_n_neg}."
        neg_idxs = np.random.choice(n_neg, sel_n_neg, replace=False)
        neg_idxs = np.nonzero(a_list < 1)[0][neg_idxs].tolist()

        idxs = pos_idxs + neg_idxs
        new_pos_rate = self.stat_one_attr(attr_name, idxs)
        # print(f"New pos rate: {new_pos_rate:.3f}")
        assert np.isclose(new_pos_rate, pos_rate, atol=1e-3), f"Not matched pos rate. Expected: {pos_rate}, but get {new_pos_rate}"
        return idxs


class CelebA(FaceDataset, ExDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, one attribute.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: str = "attr",
            return_id=False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            make_tensor_attr=True,
            double_img=False,
            dict_item=True,
            percent=1.,
    ) -> None:
        import pandas
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        self.target_type = target_type
        self.return_attr = target_type == 'attr'

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        # bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        # landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        # self.bbox = torch.as_tensor(bbox[mask].values)
        # self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        # self.attr = torch.as_tensor(attr[mask].values)
        if make_tensor_attr:
            self.attr = torch.as_tensor(attr[mask].values)
        else:
            self.attr = attr[mask].values
        self.attr = (self.attr + 1) / 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.classes = ('Not_' + self.target_type, self.target_type)

        self.double_img = double_img
        self.return_id = return_id

        self.domain = 'CelebA'
        self.domain_id = 0
        self.dict_item = dict_item

        if percent < 1.:
            sel_idxs = np.random.choice(len(self.filename), int(percent * len(self.filename)),
                                        replace=False)
            self.filename = self.filename[sel_idxs]
            self.identity = self.identity[sel_idxs]
            self.attr = self.attr[sel_idxs]

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int):
        img = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        attrs = self.attr[index, :]
        if self.target_type == 'none':
            target = -1
        elif self.target_type != 'attr':
            target = attrs[self.attr_names.index(self.target_type)]
        else:
            target = attrs
        # attrs = {a: v for a, v in zip(self.attr_names, attrs)}

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.double_img:
            return (img, img, attrs) if self.return_attr else (img, img, target)
        else:
            if self.dict_item:
                assert not self.return_attr
                assert not self.return_id
                return {'image': img, 'target': target.item(), 'domain': 0}
            else:
                if self.return_attr:
                    if self.return_id:
                        return img, attrs, self.identity[index]
                    else:
                        return img, attrs
                else:
                    # return img, target
                    return img, int(target.item())  # FIXME ad-hoc for simclr

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    def stat_attr(self, idxs=None, verbose=True):
        df_dict = {"attr_name": [], "-": [], "+": [], "pos_rate": []}
        for ai, a in enumerate(self.attr_names):
            if idxs is None:
                a_list = self.attr[:, ai]
            else:
                a_list = self.attr[idxs, ai]
            a_list = a_list.numpy()
            df_dict['attr_name'].append(a)
            df_dict['+'].append(np.sum(a_list > 0))
            df_dict['-'].append(np.sum(a_list < 1))
            df_dict['pos_rate'].append(np.sum(a_list > 0) / len(a_list))
        df = pandas.DataFrame(df_dict).set_index('attr_name')
        if verbose:
            print(df)
        return df

    def stat_one_attr(self, attr_name, indexes=None):
        ai = self.attr_names.index(attr_name)
        a_list = self.attr[:, ai].numpy()
        if indexes is not None:
            a_list = a_list[indexes]
        n_pos = np.sum(a_list > 0)
        n_neg = np.sum(a_list < 1)
        pos_rate = n_pos * 1. / len(a_list)
        return pos_rate

    def bias_attr(self, attr_name, n_sample, pos_rate):
        """Resample the attributes."""
        ai = self.attr_names.index(attr_name)
        a_list = self.attr[:, ai].numpy()
        n_pos = np.sum(a_list > 0)
        n_neg = np.sum(a_list < 1)

        assert n_sample <= len(self), "Not enough samples."
        sel_n_pos = int(n_sample * pos_rate)
        assert sel_n_pos < n_pos, f"Not enough positive samples. Have {n_pos} pos samples," \
                                  f" but wanted {sel_n_pos}."
        pos_idxs = np.random.choice(n_pos, sel_n_pos, replace=False)
        pos_idxs = np.nonzero(a_list > 0)[0][pos_idxs].tolist()
        sel_n_neg = int(n_sample * (1 - pos_rate))
        assert sel_n_neg < n_neg, f"Not enough positive samples. Have {n_neg} pos samples," \
                                  f" but wanted {sel_n_neg}."
        neg_idxs = np.random.choice(n_neg, sel_n_neg, replace=False)
        neg_idxs = np.nonzero(a_list < 1)[0][neg_idxs].tolist()

        idxs = pos_idxs + neg_idxs
        new_pos_rate = self.stat_one_attr(attr_name, idxs)
        # print(f"New pos rate: {new_pos_rate:.3f}")
        assert np.isclose(new_pos_rate, pos_rate, atol=1e-3), f"Not matched pos rate. Expected: {pos_rate}, but get {new_pos_rate}"
        return idxs

    def get_all_targets(self, indices):
        return self.targets[indices]

    @property
    def targets(self):
        return self.attr[:, self.attr_names.index(self.target_type)].numpy()


class iCartoonFace(FaceDataset):
    base_folder = "icartoon"
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("1USvdrXUExzuB1O5z0nDpR74pDNzI0KxL", None, "personai_icartoonface_rectrain.zip"),
        ("1lUq5-BgNgqj-gIP33XLiLyJL10gQupza", None, "personai_icartoonface_rectest.zip"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: str = "attr",
            return_id=False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            make_tensor_attr=True,
            double_img=False,
    ) -> None:
        import pandas
        super(iCartoonFace, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.split = split
        self.target_type = target_type
        self.return_attr = target_type == 'attr'

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        raise NotImplementedError('not finished impl')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        # bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        # landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        # self.bbox = torch.as_tensor(bbox[mask].values)
        # self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        # self.attr = torch.as_tensor(attr[mask].values)
        if make_tensor_attr:
            self.attr = torch.as_tensor(attr[mask].values)
        else:
            self.attr = attr[mask].values
        self.attr = (self.attr + 1) / 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.classes = ('Not_' + self.target_type, self.target_type)

        self.double_img = double_img
        self.return_id = return_id

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int):
        img = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        attrs = self.attr[index, :]
        if self.target_type != 'attr':
            target = attrs[self.attr_names.index(self.target_type)]
        else:
            target = attrs
        # attrs = {a: v for a, v in zip(self.attr_names, attrs)}

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.double_img:
            return (img, img, attrs) if self.return_attr else (img, img, target)
        else:
            if self.return_attr:
                if self.return_id:
                    return img, attrs, self.identity[index]
                else:
                    return img, attrs
            else:
                return img, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class _LFW(FaceDataset):
    base_folder = 'lfw-py'
    download_url_prefix = "http://vis-www.cs.umass.edu/lfw/"
    attr_url = ("https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt", "lfw_attributes.txt", "")

    file_dict = {
        'original': ("lfw", "lfw.tgz", "a17d05bd522c52d84eca14327a23d494"),
        'funneled': ("lfw_funneled", "lfw-funneled.tgz", "1b42dfed7d15c9b2dd63d5e5840c86ad"),
        'deepfunneled': ("lfw-deepfunneled", "lfw-deepfunneled.tgz", "68331da3eb755a505a502b5aacb3c201"),
    }
    checksums = {
        'pairs.txt': '9f1ba174e4e1c508ff7cdf10ac338a7d',
        'pairsDevTest.txt': '5132f7440eb68cf58910c8a45a2ac10b',
        'pairsDevTrain.txt': '4f27cbf15b2da4a85c1907eb4181ad21',
        'people.txt': '450f0863dd89e85e73936a6d71a3474b',
        'peopleDevTest.txt': 'e4bf5be0a43b5dcd9dc5ccfcb8fb19c5',
        'peopleDevTrain.txt': '54eaac34beb6d042ed3a7d883e247a21',
        'lfw-names.txt': 'a6d0a479bd074669f656265a6e693f6d'
    }
    annot_file = {'10fold': '', 'train': 'DevTrain', 'test': 'DevTest'}
    names = "lfw-names.txt"

    def __init__(
            self,
            root: str,
            split: str,
            image_set: str,
            view: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super(_LFW, self).__init__(os.path.join(root, self.base_folder),
                                   transform=transform, target_transform=target_transform)

        self.image_set = verify_str_arg(image_set.lower(), 'image_set', self.file_dict.keys())
        images_dir, self.filename, self.md5 = self.file_dict[self.image_set]

        self.view = verify_str_arg(view.lower(), 'view', ['people', 'pairs'])
        self.split = verify_str_arg(split.lower(), 'split', ['10fold', 'train', 'test'])
        self.labels_file = f"{self.view}{self.annot_file[self.split]}.txt"
        self.data: List[Any] = []

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.images_dir = os.path.join(self.root, images_dir)

    def _loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _check_integrity(self):
        st1 = check_integrity(os.path.join(self.root, self.filename), self.md5)
        st2 = check_integrity(os.path.join(self.root, self.labels_file), self.checksums[self.labels_file])
        if not st1 or not st2:
            return False
        if self.view == "people":
            return check_integrity(os.path.join(self.root, self.names), self.checksums[self.names])
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        url = f"{self.download_url_prefix}{self.filename}"
        download_and_extract_archive(url, self.root, filename=self.filename, md5=self.md5)
        download_url(f"{self.download_url_prefix}{self.labels_file}", self.root)
        download_url(self.attr_url[0], self.root)
        if self.view == "people":
            download_url(f"{self.download_url_prefix}{self.names}", self.root)

    def _get_path(self, identity, no):
        return os.path.join(self.images_dir, identity, f"{identity}_{int(no):04d}.jpg")

    def extra_repr(self) -> str:
        return f"Alignment: {self.image_set}\nSplit: {self.split}"

    def __len__(self):
        return len(self.data)


class LFWPeople(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset for face identification.

    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold`` (default).
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
            self,
            root: str,
            split: str = "10fold",
            image_set: str = "funneled",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super(LFWPeople, self).__init__(root, split, image_set, "people",
                                        transform, target_transform, download)

        self.class_to_idx = self._get_classes()
        self.data, self.targets = self._get_people()

    def _get_people(self):
        data, targets = [], []
        with open(os.path.join(self.root, self.labels_file), 'r') as f:
            lines = f.readlines()
            n_folds, s = (int(lines[0]), 1) if self.split == "10fold" else (1, 0)

            for fold in range(n_folds):
                n_lines = int(lines[s])
                people = [line.strip().split("\t") for line in lines[s + 1: s + n_lines + 1]]
                s += n_lines + 1
                for i, (identity, num_imgs) in enumerate(people):
                    for num in range(1, int(num_imgs) + 1):
                        img = self._get_path(identity, num)
                        data.append(img)
                        targets.append(self.class_to_idx[identity])

        return data, targets

    def _get_classes(self):
        with open(os.path.join(self.root, self.names), 'r') as f:
            lines = f.readlines()
            names = [line.strip().split()[0] for line in lines]
        class_to_idx = {name: i for i, name in enumerate(names)}
        return class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the identity of the person.
        """
        img = self._loader(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self) -> str:
        return super().extra_repr() + "\nClasses (identities): {}".format(len(self.class_to_idx))


class LFWAttr(_LFW):

    def __init__(
            self,
            root: str,
            split: str = "train",
            image_set: str = "funneled",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            target_type: str = 'attr',
    ):
        super(LFWAttr, self).__init__(root, split, image_set, "people",
                                      transform, target_transform, download)

        self.identity_to_idx = self._get_identities()
        self.identity_to_attrs, self.attr_names = self._get_attrs()
        self.data, self.identities, self.attr = self._get_people()
        self.target_type = target_type
        if target_type == 'attr':
            self.attr = (self.attr > 0).astype('int')
        elif target_type == 'attr_score':
            pass
        else:
            raise NotImplementedError(f"target_type: {target_type}")
        self.attr = torch.as_tensor(self.attr)

    def _get_attrs(self):
        with open(os.path.join(self.root, self.attr_url[1]), 'r') as f:
            f.readline()
            columns = f.readline()
            columns = columns[2:-1].split('\t')
        df = pandas.read_csv(os.path.join(self.root, self.attr_url[1]), skiprows=2, names=columns,
                             delimiter='\t', index_col=['person', 'imagenum'])
        # print(df)  # df.loc['Aaron Eckhart']
        return df, df.columns.values.tolist()

    def _get_people(self):
        data, id_idxs, attrs = [], [], []
        attr_missing_identity = defaultdict(int)
        # all_attr_identities = self.identity_to_attrs.reset_index()['person'].values
        with open(os.path.join(self.root, self.labels_file), 'r') as f:
            lines = f.readlines()
            n_folds, s = (int(lines[0]), 1) if self.split == "10fold" else (1, 0)

            for fold in range(n_folds):
                n_lines = int(lines[s])
                people = [line.strip().split("\t") for line in lines[s + 1: s + n_lines + 1]]
                s += n_lines + 1
                for i, (identity, num_imgs) in enumerate(people):
                    for num in range(1, int(num_imgs) + 1):
                        if identity in attr_missing_identity or (identity.replace('_', ' '), num) not in self.identity_to_attrs.index:
                            attr_missing_identity[identity] += 1
                        else:
                            attrs.append(self.identity_to_attrs.loc[(identity.replace('_', ' '), num)].values)
                            img = self._get_path(identity, num)
                            data.append(img)
                            id_idxs.append(self.identity_to_idx[identity])
        print(f"Missing attributes: {sum([v for _, v in attr_missing_identity.items()])} faces "
              f"from {len(attr_missing_identity)} people")
        attrs = np.stack(attrs, axis=0)
        return data, id_idxs, attrs

    def _get_identities(self):
        with open(os.path.join(self.root, self.names), 'r') as f:
            lines = f.readlines()
            names = [line.strip().split()[0] for line in lines]
        identity_to_idx = {name: i for i, name in enumerate(names)}
        return identity_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the attributes of the face.
        """
        img = self._loader(self.data[index])
        target = self.attr[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def stat_attr(self, idxs=None, verbose=True):
        df_dict = {"attr_name": [], "-": [], "+": [], "pos_rate": []}
        for ai, a in enumerate(self.attr_names):
            if idxs is None:
                a_list = self.attr[:, ai]
            else:
                a_list = self.attr[idxs, ai]
            a_list = a_list.numpy()
            df_dict['attr_name'].append(a)
            df_dict['+'].append(np.sum(a_list > 0))
            df_dict['-'].append(np.sum(a_list < 1))
            df_dict['pos_rate'].append(np.sum(a_list > 0) / len(a_list))
        df = pandas.DataFrame(df_dict).set_index('attr_name')
        if verbose:
            print(df)
        return df

    def stat_one_attr(self, attr_name, indexes=None):
        ai = self.attr_names.index(attr_name)
        a_list = self.attr[:, ai].numpy()
        if indexes is not None:
            a_list = a_list[indexes]
        n_pos = np.sum(a_list > 0)
        n_neg = np.sum(a_list < 1)
        pos_rate = n_pos * 1. / len(a_list)
        return pos_rate

    def bias_attr(self, attr_name, n_sample, pos_rate):
        """Resample the attributes."""
        ai = self.attr_names.index(attr_name)
        a_list = self.attr[:, ai].numpy()
        n_pos = np.sum(a_list > 0)
        n_neg = np.sum(a_list < 1)

        assert n_sample <= len(self), "Not enough samples."
        sel_n_pos = int(n_sample * pos_rate)
        assert sel_n_pos < n_pos, f"Not enough positive samples. Have {n_pos} pos samples," \
                                  f" but wanted {sel_n_pos}."
        pos_idxs = np.random.choice(n_pos, sel_n_pos, replace=False)
        pos_idxs = np.nonzero(a_list > 0)[0][pos_idxs].tolist()
        sel_n_neg = int(n_sample * (1 - pos_rate))
        assert sel_n_neg < n_neg, f"Not enough positive samples. Have {n_neg} pos samples," \
                                  f" but wanted {sel_n_neg}."
        neg_idxs = np.random.choice(n_neg, sel_n_neg, replace=False)
        neg_idxs = np.nonzero(a_list < 1)[0][neg_idxs].tolist()

        idxs = pos_idxs + neg_idxs
        new_pos_rate = self.stat_one_attr(attr_name, idxs)
        # print(f"New pos rate: {new_pos_rate:.3f}")
        assert np.isclose(new_pos_rate, pos_rate, atol=1e-3), f"Not matched pos rate. Expected: {pos_rate}, but get {new_pos_rate}"
        return idxs


def get_joint_conditional_mask(ds: CelebA, SA_idx: int, A_idx: int):
    if isinstance(ds, IndexedDataset):
        ds = ds.dataset
    all_attr = []
    dl = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4)
    for _, attrs in tqdm(dl, desc='load attrs', leave=False):
        all_attr.append(attrs.cpu().detach())
    all_attr = torch.cat(all_attr, dim=0).cpu().numpy()

    cond_dict = {}
    for A_val in [0, 1]:
        for SA_val in [0, 1]:
            cond_dict[f"SA{SA_val},A{A_val}"] = np.argwhere((all_attr[:, SA_idx] == SA_val) & (all_attr[:, A_idx] == A_val))
    return cond_dict


def get_conditional_mask(ds: CelebA, A_idx: int):
    if isinstance(ds, IndexedDataset):
        ds = ds.dataset
    all_attr = []
    dl = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4)
    for _, attrs in tqdm(dl, desc='load attrs', leave=False):
        all_attr.append(attrs.cpu().detach())
    all_attr = torch.cat(all_attr, dim=0).cpu().numpy()

    cond_dict = {}
    for A_val in [0, 1]:
        cond_dict[f"A{A_val}"] = np.argwhere((all_attr[:, A_idx] == A_val))
    return cond_dict


def get_fea_A_SA(ds, model, A_idx, SA_idx, device):
    if isinstance(ds, IndexedDataset):
        ds = ds.dataset
    all_attr, all_fea = [], []
    dl = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4)
    for images, attrs in tqdm(dl, desc='load attrs', leave=False):
        features = model.encode(images.to(device))
        all_fea.append(features.cpu().detach())
        all_attr.append(attrs.cpu().detach())
    all_fea = torch.cat(all_fea, dim=0).cpu().numpy()
    all_attr = torch.cat(all_attr, dim=0).cpu().numpy()

    labels = all_attr[:, A_idx]
    groups = all_attr[:, SA_idx]
    return labels, groups, all_fea


def stat_dataset(ds):
    n = len(ds)

    for i in [0, 12, 43]:
        if isinstance(ds, (CelebA, LFWAttr, UTKFace, FairFace)):
            ds.return_attr = True
            img, attrs = ds[i]
        else:
            img, target, attrs = ds[i]
        # img, attrs = ds[i]
        print(
            f"[{i:3d}] img shape {img.shape}, attrs {attrs}")
    print()
    stat = {
        f"#sample": n,
        "image shape": img.shape,
        "target type": ds.target_type,
    }
    for k, v in stat.items():
        print(f"{k:>20s}: {v}")
    ds.stat_attr()


if __name__ == '__main__':
    parser = ArgumentParser(prog='dataset', description='Download datasets.')
    parser.add_argument('--data', type=str, default='IJB')
    args = parser.parse_args()

    if args.data == 'IJB':
        ds = IJB(data_root, download=True)
        stat_dataset(ds)
        # NOTE not fully implemented because the dataset does not include attributes or social group labels.
    elif args.data == 'CelebA':
        ds = CelebA(data_root, transform=trns.ToTensor(), download=False)
        # If zip file error occurs, try to download ALL files from google driver directly.
        stat_dataset(ds)
    elif args.data == 'UTKFace':  # for fine-tune where we need to adjust the demorgraphic disparaty.
        ds = UTKFace(data_root, transform=trns.ToTensor())
        stat_dataset(ds)
    elif args.data == 'FairFace':  # for pre-train
        ds = FairFace(data_root, transform=trns.ToTensor(), download=True)
        stat_dataset(ds)
    elif args.data == 'LFW':  # for pre-train
        ds = LFWAttr(data_root, transform=trns.ToTensor(), download=True)
        stat_dataset(ds)
    else:
        raise RuntimeError(f"Invalid data: {args.data}")

# TODO: RFW test set (for race-aware) https://urldefense.com/v3/__https://drive.google.com/drive/folders/1a2hy9qSF_LIJuqL-KzxcH-e6tMAL_UGd?usp=sharing__;!!HXCxUKc!jVFqtDIE8yJXq_lwULYOrzDEEsQITobel1TwOpgswLsUo0Xl284O1f0VkS-50zCJ3Q$
# TODO: MS1M remove RFW MS1M_wo_RFW_index.zip http://www.whdeng.cn/RFW/download/MS1M_wo_RFW_index.zip
