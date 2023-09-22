import bisect
import copy
import json
import os
import random
from abc import ABC
from collections import defaultdict
from typing import Dict
from typing import List

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torchvision import transforms
from utils.ssl.datasets.augmentation.randaugment import RandAugment


class ExDataset(Dataset, ABC):
    """Extended dataset."""
    classes = ['']  # to override

    def __init__(self, split):
        self.split = split

    def get_all_targets(self, indices):
        # return self.targets[indices]
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class NoisySemiSupDataset(Dataset):
    """Semi-supervised dataset with noisy labels."""

    def __init__(self, dataset: ExDataset, mode, transform, client_ratio=0., noise_label=None,
                 # pred=np.empty(0), probability=np.empty(0),
                 sample_weights=None,
                 noise_ratio=0., noise_mode='asym', noise_file='', new_noise_file=True,
                 seed=42, ood_dataset: ExDataset = None, include_id_to_ood=True):
        # super(DictDataset, self).__init__()
        self.dataset = dataset
        self.ood_dataset = ood_dataset  # all will be used as unlabeled
        self.mode = mode
        self.transform = transform
        self.include_id_to_ood = include_id_to_ood

        # self.noise_ratio = noise_ratio
        transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                      8: 8}  # class transition for asymmetric noise
        self.dataset_indices = np.arange(len(dataset))
        self.id_data_end_idx = np.max(self.dataset_indices)  # include the last
        # ensure that ID and OoD indeces are not overlapped.
        # note that ID data will be subsampled.
        if self.ood_dataset is not None:
            ood_idxs = self.id_data_end_idx + 1 + np.arange(len(self.ood_dataset))

        if self.mode == 'test':
            assert dataset.split == 'test'
            if self.ood_dataset is not None and len(self.ood_dataset) > 0:
                self.dataset_indices = np.concatenate([self.dataset_indices,
                                                       ood_idxs], axis=0)
                # self.ood_dataset_indices = np.arange(len(self.ood_dataset))
            # else:
            #     self.ood_dataset_indices = []
        else:
            assert dataset.split == 'train'
            # train_data, train_label = data, targets

            # split ID dataset into client and server
            self.dataset_indices = self.split_client_server_data(
                self.mode, client_ratio, seed, self.dataset_indices)
            train_label = dataset.get_all_targets(self.dataset_indices)

            if self.mode != 'client' and self.ood_dataset is not None \
                    and len(self.ood_dataset) > 0:
                if self.include_id_to_ood:
                    # include splitted ID data into OoD dataset
                    self.dataset_indices = np.concatenate([
                        self.dataset_indices, ood_idxs], axis=0)
                    train_label = np.concatenate([
                        train_label,
                        self.ood_dataset.get_all_targets(np.arange(len(self.ood_dataset)))], axis=0)
                else:
                    self.dataset_indices = ood_idxs
                    train_label = self.ood_dataset.get_all_targets(np.arange(len(self.ood_dataset)))

            if self.mode == 'client':
                assert client_ratio > 0, f"Not set client data as client_ratio={client_ratio}"
                # self.train_data = train_data
                # self.train_label = train_label
            elif self.mode == 'query':
                # assert client_ratio > 0, f"Not set client data as client_ratio={client_ratio}"
                pass
                # self.train_data = train_data
                # self.train_label = train_label
            else:  # data for training
                if noise_label is None:
                    if noise_ratio > 0:
                        # make synthetic noise label
                        if os.path.exists(noise_file):
                            # print(f"load noise file from {noise_file}")
                            noise_label = json.load(open(noise_file, "r"))
                        elif new_noise_file:  # inject noise
                            print(f"not found noise file. To generate new")
                            noise_label = make_noise_labels(
                                noise_ratio, len(dataset.classes), noise_mode,
                                train_label, class_transition=transition)
                            print("save noisy labels to %s ..." % noise_file)
                            if not os.path.exists(os.path.dirname(noise_file)):
                                os.makedirs(os.path.dirname(noise_file))
                            json.dump(noise_label, open(noise_file, "w"))
                        else:
                            raise RuntimeError(f"Not found noise_file at {noise_file} and did not create new.")
                    else:
                        noise_label = train_label
                if sample_weights is None:
                    sample_weights = np.ones(len(noise_label))
                assert len(noise_label) == len(train_label), \
                    f"Inconsistent label num: expected {len(train_label)}, but get" \
                    f" {len(noise_label)} noise labels."

                noise_labeled_idxs = np.where(np.array(noise_label) >= 0)[0]
                in_dist_unlabeled_idxs = np.where(np.array(noise_label) == -1)[0]
                out_dist_unlabeled_idxs = np.where(np.array(noise_label) == -2)[0]

                if self.mode == 'all':  # all data including labeled or unlabeled.
                    pass
                    # self.train_data = train_data
                    # self.noise_label = noise_label
                elif self.mode == 'noise_labeled':  # all noise labeled
                    # self.noise_labeled_idxs = noise_labeled_idxs
                    # self.train_data = train_data[noise_labeled_idxs]
                    self.noise_label = [noise_label[i] for i in noise_labeled_idxs]
                    self.sample_weights = [sample_weights[i] for i in noise_labeled_idxs]
                    self.dataset_indices = self.dataset_indices[noise_labeled_idxs]
                else:  # labeled or unlabeled
                    if len(noise_labeled_idxs) == len(noise_label):  # fully labeled.
                        # if pred is not None and len(pred) > 0:
                        #     if self.mode == "labeled":
                        #         pred_idx = pred.nonzero()[0]
                        #         # self.probability = [probability[i] for i in pred_idx]
                        #         #
                        #         # clean = (np.array(noise_label) == np.array(train_label))
                        #         # auc_meter = AUCMeter()
                        #         # auc_meter.reset()
                        #         # auc_meter.add(probability, clean)
                        #         # auc, _, _ = auc_meter.value()
                        #         # print('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                        #         # wandb.log({'clean label auc': auc, 'num labeled': pred.sum()}, commit=False)
                        #
                        #     elif self.mode == "unlabeled":
                        #         pred_idx = (1 - pred).nonzero()[0]
                        #     else:
                        #         raise ValueError(f"Unexpected mode: {self.mode}")
                        #
                        #     # self.train_data = train_data[pred_idx]
                        #     # self.noise_label = [noise_label[i] for i in pred_idx]
                        #
                        #     self.noise_label = [noise_label[i] for i in pred_idx]
                        #     self.dataset_indices = self.dataset_indices[pred_idx]
                        # else:
                        if self.mode == "labeled":
                            self.noise_label = [noise_label[i] for i in noise_labeled_idxs]
                            self.sample_weights = [sample_weights[i] for i in noise_labeled_idxs]
                            self.dataset_indices = self.dataset_indices[noise_labeled_idxs]
                        elif self.mode == "unlabeled":
                            self.noise_label = []
                            self.sample_weights = []
                            self.dataset_indices = []
                        elif self.mode == "ood":
                            self.noise_label = []
                            self.sample_weights = []
                            self.dataset_indices = []

                        print("%s data has a size of %d" % (self.mode, len(self.noise_label)))
                        if self.sample_weights is not None and len(self.sample_weights) > 0:
                            print(f" sample weights: max: {np.max(self.sample_weights)}, min: {np.min(self.sample_weights)}, "
                                  f"mean: {np.mean(self.sample_weights)}, median: {np.median(self.sample_weights)}")
                    else:
                        if self.mode == "labeled":
                            # if pred is not None and len(pred) > 0:
                            #     pred_idx = pred.nonzero()[0]
                            #     self.probability = [probability[i] for i in pred_idx]
                            #
                            #     clean = (np.array(noise_label)[noise_labeled_idxs] == np.array(train_label)[noise_labeled_idxs])
                            #     auc_meter = AUCMeter()
                            #     auc_meter.reset()
                            #     auc_meter.add(probability, clean)
                            #     auc, _, _ = auc_meter.value()
                            #     print(' Numer of labeled samples:%d   AUC:%.3f' % (pred.sum(), auc))
                            #     wandb.log({'clean label auc': auc, 'num labeled': pred.sum()}, commit=False)
                            # else:
                            #     pred_idx = np.arange(len(noise_labeled_idxs))
                            #     self.probability = [1. for _ in pred_idx]
                            pred_idx = np.arange(len(noise_labeled_idxs))

                            # self.train_data = train_data[noise_labeled_idxs[pred_idx]]
                            # self.noise_label = [noise_label[noise_labeled_idxs[i]] for i in
                            #                     pred_idx]

                            self.noise_label = [noise_label[noise_labeled_idxs[i]] for i in pred_idx]
                            self.sample_weights = [sample_weights[i] for i in pred_idx]
                            self.dataset_indices = self.dataset_indices[noise_labeled_idxs[pred_idx]]

                        elif self.mode == "unlabeled":
                            # if pred is not None and len(pred) > 0:
                            #     pred_idx = (1 - pred).nonzero()[0]
                            # else:
                            #     pred_idx = []
                            pred_idx = []

                            all_ulab_idxs = np.concatenate([noise_labeled_idxs[pred_idx],
                                                            in_dist_unlabeled_idxs])
                            # self.train_data = train_data[all_ulab_idxs]
                            self.noise_label = None
                            self.sample_weights = [sample_weights[i] for i in all_ulab_idxs]
                            self.dataset_indices = self.dataset_indices[all_ulab_idxs]
                        elif self.mode == "ood":
                            # self.train_data = train_data[all_ulab_idxs]
                            self.noise_label = None
                            self.sample_weights = None  # FIXME shall we add weights?
                            self.dataset_indices = self.dataset_indices[out_dist_unlabeled_idxs]
                        else:
                            raise ValueError(f"Unexpected mode: {self.mode}")

                        print(" %s data has a size of %d" % (self.mode, len(self.dataset_indices)))
                        if self.sample_weights is not None and len(self.sample_weights) > 0:
                            print(f" sample weights: max: {np.max(self.sample_weights)}, min: {np.min(self.sample_weights)}, "
                                  f"mean: {np.mean(self.sample_weights)}, median: {np.median(self.sample_weights)}")

    def split_client_server_data(self, mode, client_ratio, seed, indices):
        if client_ratio > 0:
            idxs = np.arange(len(indices))
            rng = np.random.RandomState(seed)
            rng.shuffle(idxs)
            n_clt = int(client_ratio * len(indices))
            if mode == 'client':
                idxs = idxs[:n_clt]
            else:
                idxs = idxs[n_clt:]
            # train_data = train_data[idxs]
            # train_label = [train_label[i] for i in idxs]
            indices = indices[idxs]
        return indices

    # @property
    # def id_data_num(self):
    #     """num of in-distribution samples"""
    #     return len(self.dataset_indices)

    def __getitem__(self, index):
        data_dict = {}
        pre_data_dict = self.get_img_target(index)
        img = pre_data_dict['image']
        target = pre_data_dict['target']
        if self.mode == 'labeled':
            data_dict['image'] = self.transform(img)
            data_dict['target'] = self.noise_label[index]
            data_dict['weight'] = self.sample_weights[index]
            # data_dict['prob'] = self.probability[index]
            data_dict['true target'] = target
        elif self.mode == 'unlabeled':
            data_dict['image'] = self.transform(img)
            data_dict['true target'] = target
            data_dict['weight'] = self.sample_weights[index]
        elif self.mode == 'ood':
            data_dict['image'] = self.transform(img)
            if 'domain' in pre_data_dict:
                data_dict['domain'] = pre_data_dict['domain']
        elif self.mode in ['all', 'noise_labeled']:
            data_dict['image'] = self.transform(img)
            data_dict['target'] = self.noise_label[index]
            data_dict['weight'] = self.sample_weights[index]
            data_dict['index'] = index
        elif self.mode in ['test', 'query', 'client']:
            data_dict['image'] = self.transform(img)
            data_dict['target'] = target
            if 'domain' in pre_data_dict:
                data_dict['domain'] = pre_data_dict['domain']
        return data_dict

    def get_img_target(self, index):
        _idx = self.dataset_indices[index]
        if self.is_id_idx(_idx):
            data_dict = self.dataset[_idx]
        else:
            data_dict = self.ood_dataset[_idx - 1 - self.id_data_end_idx]
        return data_dict

    def is_id_idx(self, idx):
        """Is in-distribution index"""
        return idx <= self.id_data_end_idx

    def __len__(self):
        return len(self.dataset_indices)  # + len(self.ood_dataset_indices)

    # def update_sample_weights(self, new_weights, normalize=True):
    #     new_weights = copy.deepcopy(new_weights)
    #     print(f"Update sample weights: max: {np.max(new_weights)}, min: {np.min(new_weights)}, "
    #           f"mean: {np.mean(new_weights)}, median: {np.median(new_weights)}")
    #     if normalize:
    #         new_weights = new_weights / np.sum(new_weights)
    #     self.sample_weights = new_weights


class DivideMixDataset(NoisySemiSupDataset):
    def __getitem__(self, index):
        data_dict = {}
        if self.mode == 'unlabeled':
            pre_data_dict = self.get_img_target(index)
            img = pre_data_dict['image']
            data_dict['image0'] = self.transform(img)
            data_dict['image1'] = self.transform(img)
            data_dict['true target'] = pre_data_dict['target']
        elif self.mode == 'labeled':
            pre_data_dict = self.get_img_target(index)
            img = pre_data_dict['image']
            data_dict['image0'] = self.transform(img)
            data_dict['image1'] = self.transform(img)
            data_dict['target'] = self.noise_label[index]
            data_dict['prob'] = self.probability[index]
            data_dict['true target'] = pre_data_dict['target']
        else:
            return super(DivideMixDataset, self).__getitem__(index)
        return data_dict


class FixMatchDataset(NoisySemiSupDataset):
    def __init__(self, dataset: ExDataset, mode, transform, **kwargs):
        super(FixMatchDataset, self).__init__(dataset, mode, transform, **kwargs)
        strong_transform = transforms.Compose([RandAugment(3, 5), copy.deepcopy(transform)])
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        data_dict = {}
        if self.mode == 'unlabeled':
            pre_data_dict = self.get_img_target(index)
            img = pre_data_dict['image']
            data_dict['image_w'] = self.transform(img)
            data_dict['image_s'] = self.strong_transform(img)
            data_dict['true target'] = pre_data_dict['target']
            data_dict['weight'] = self.sample_weights[index]
            data_dict['index'] = index
        elif self.mode == 'labeled':
            pre_data_dict = self.get_img_target(index)
            img = pre_data_dict['image']
            data_dict['image'] = self.transform(img)
            data_dict['target'] = self.noise_label[index]
            data_dict['weight'] = self.sample_weights[index]
            data_dict['true target'] = pre_data_dict['target']
        else:
            return super(FixMatchDataset, self).__getitem__(index)
        return data_dict


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def make_noise_labels(noise_ratio, num_classes, noise_mode, train_label, seed=42,
                      class_transition: Dict = None):
    if noise_ratio == 0.:
        return train_label.copy()
    if class_transition is None:
        class_transition = dict()
    num_samples = len(train_label)
    rng = np.random.RandomState(seed)
    noise_label = []
    idx = list(range(num_samples))
    random.shuffle(idx)
    num_noise = int(noise_ratio * num_samples)
    noise_idx = idx[:num_noise]
    for i in range(num_samples):
        if i in noise_idx:
            if noise_mode == 'sym':
                noiselabel = rng.randint(num_classes)
                noise_label.append(noiselabel)
            elif noise_mode == 'asym':
                noiselabel = class_transition[train_label[i]]
                noise_label.append(noiselabel)
        else:
            noise_label.append(train_label[i])
    return noise_label


class CatExDataset(ExDataset):
    def __init__(self, datasets: List[ExDataset]):
        super(CatExDataset, self).__init__(split=datasets[0].split)
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

    datasets: List[ExDataset]
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
    def check_consistency(datasets: List[ExDataset]):
        if len(datasets) == 1:
            return
        else:
            for ds in datasets[1:]:
                assert ds.split == datasets[0].split
            u_domains = np.unique([ds.domain for ds in datasets])
            assert len(u_domains) == len(datasets), f"Found duplicated domains: {u_domains}"
