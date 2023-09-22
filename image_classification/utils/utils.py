# from utils.mmd import mix_rbf_mmd2
import random
import warnings

import copy
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # rng = np.random.RandomState(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class BestMeter:
    """Get the best in stream."""

    def __init__(self, best_mode='max', delay=0):
        self.mode = best_mode
        if self.mode == 'max':
            self.best_value = -np.inf
        elif self.mode == 'min':
            self.best_value = np.inf
        else:
            raise NotImplementedError(f"mode: {self.mode}")
        self._best_changed = False
        self._cnt = 0
        self.delay = delay

    @property
    def best_changed(self):
        return self._best_changed

    def add(self, value):
        self._cnt += 1
        if self.delay > 0 and self._cnt < self.delay:
            self._best_changed = False
            return
        if self.mode == 'max':
            self._best_changed = value >= self.best_value
        if self.mode == 'min':
            self._best_changed = value <= self.best_value
        if self._best_changed:
            self.best_value = value

    def state_dict(self):
        return {'mode': self.mode, 'best_value': self.best_value}

    def load_state_dict(self, state_dict):
        self.mode = state_dict['mode']
        self.best_value = state_dict['best_value']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    def extend(self, vals):
        self.values.extend(vals)
        self.counter += len(vals)

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        if len(self.values) > 0:
            return sum(self.values) / len(self.values)
        else:
            return np.nan

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


def np_balanced_accuracy(y_true, y_pred, sample_weight=None):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class_acc)):
        warnings.warn("y_pred contains classes not in y_true")
        per_class_acc = per_class_acc[~np.isnan(per_class_acc)]
    bacc = np.mean(per_class_acc)
    return bacc, per_class_acc


def Acc(targets, preds):
    '''
    PyTorch operation: Accuracy.

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        acc: float
    '''
    correct = preds.eq(targets.view_as(preds)).sum().item()
    total = torch.numel(preds)
    acc = correct / total
    return acc


def FPR(targets, preds):
    '''
    PyTorch operation: False positive rate.

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        FPR: float
    '''
    N = (targets == 0).sum().item()  # negative sample number
    if N == 0:
        return -1, N
    FP = torch.logical_and(targets == 0, preds.squeeze() == 1).sum().item()  # FP sample number
    FPR = FP / N
    return FPR, N


def FNR(targets, preds):
    '''
    PyTorch operation: False negative rate.

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        FNR: float
    '''
    P = (targets == 1).sum().item()  # positive sample number
    if P == 0:
        return -1, P
    FN = torch.logical_and(targets == 1, preds.squeeze() == 0).sum().item()  # FP sample number
    FNR = FN / P
    return FNR, P


def F1score(targets, preds):
    TP = torch.logical_and(targets == 1, preds.squeeze() == 1).sum().item()  # TP sample number
    TN = torch.logical_and(targets == 0, preds.squeeze() == 0).sum().item()  # TN sample number
    FP = torch.logical_and(targets == 0, preds.squeeze() == 1).sum().item()  # FP sample number
    FN = torch.logical_and(targets == 1, preds.squeeze() == 0).sum().item()  # FP sample number
    F1score = TP / (TP + 0.5 * FP + 0.5 * FN)
    return F1score


def delta_DP(preds_g0, preds_g1):
    '''
    Args:
        preds_g0: Tensor. Predictions on data from group 0.
        preds_g1: Tensor. Predictions on data from group 1.
    '''
    delta_DP = torch.abs(torch.mean(preds_g0.float()) - torch.mean(preds_g1.float()))
    return delta_DP


def delta_EOpp(targets_g0, preds_g0, targets_g1, preds_g1):
    '''
    Args:
        targets_g0: Tensor. Ground truth targets of data from group 0.
        preds_g0: Tensor. Predictions on data from group 0.
    '''
    FPR_g0 = FPR(targets_g0, preds_g0)
    FPR_g1 = FPR(targets_g1, preds_g1)

    FNR_g0 = FNR(targets_g0, preds_g0)
    FNR_g1 = FNR(targets_g1, preds_g1)

    print('FPR_g0:', FPR_g0, 'FPR_g1:', FPR_g1)
    print('FNR_g0:', FNR_g0, 'FNR_g1:', FNR_g1)

    delta_EOpp = np.abs(FPR_g0 - FPR_g1)

    return delta_EOpp


def delta_EO(targets_g0, preds_g0, targets_g1, preds_g1):
    '''
    Args:
        targets_g0: Tensor. Ground truth targets of data from group 0.
        preds_g0: Tensor. Predictions on data from group 0.
    '''
    FPR_g0 = FPR(targets_g0, preds_g0)
    FPR_g1 = FPR(targets_g1, preds_g1)

    FNR_g0 = FNR(targets_g0, preds_g0)
    FNR_g1 = FNR(targets_g1, preds_g1)

    print('FPR_g0:', FPR_g0, 'FPR_g1:', FPR_g1)
    print('FNR_g0:', FNR_g0, 'FNR_g1:', FNR_g1)

    delta_EO = np.abs(FPR_g0 - FPR_g1) + np.abs(FNR_g0 - FNR_g1)

    return delta_EO


def Gaussian_WD(x, y):
    '''
    Compute the Wasserstein distance between two groups of data x, y, assuming underlying Gaussian distributions.
    This is also used as FID in GAN.

    Args:
        x: Tensor, size=(N1,)
        y: Tensor, size=(N2,)
    '''

    var_x, mean_x = torch.var_mean(x)
    var_y, mean_y = torch.var_mean(y)

    # print('mean_x: %.2f, var_x: %.2f, mean_y: %.2f, var_y: %.2f' % (mean_x, var_x, mean_y, var_y) )

    d = (mean_x - mean_y) ** 2 + (var_x + var_y - 2 * torch.sqrt(var_x * var_y))

    return d


def LogGaussian_KLD(x, y):
    '''
    Compute the KL Divergence between two groups of data x, y, assuming underlying Log Gaussian distributions.

    Args:
        x: Tensor, size=(N1,)
        y: Tensor, size=(N2,)
    '''

    var_x, mean_x = torch.var_mean(x)
    var_y, mean_y = torch.var_mean(y)

    # print('mean_x: %.2f, var_x: %.2f, mean_y: %.2f, var_y: %.2f' % (mean_x, var_x, mean_y, var_y) )

    d = ((mean_x - mean_y) ** 2 + var_x - var_y) / (2 * var_y) + torch.log(
        torch.sqrt(var_y) / torch.sqrt(var_x))

    return d


def Exp_KLD(x, y):
    '''
    Compute the KL Divergence between two groups of data x, y, assuming underlying Exponential distributions

    Args:
        x: Tensor, size=(N1,)
        y: Tensor, size=(N2,)
    '''

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # print('mean_x: %.2f, mean_y: %.2f' % (mean_x, mean_y) )

    d = torch.log(mean_y) - torch.log(mean_x) + mean_x / mean_y - 1

    return d


# def delta_EG(sn_g0, sn_g1, sigma_list=[1, 2, 4, 8, 16]):
#     '''
#     Args:
#         sn_g0: Tensor. size=(N0,). Spectural norms of data from group 0.
#         sn_g1: Tensor. size=(N1,). Spectural norms of data from group 1.
#     '''
#
#     # calculate delta EG:
#     deltaEG = mix_rbf_mmd2(sn_g0, sn_g1, sigma_list=sigma_list)
#
#     return deltaEG


# ///////////// samplers /////////////
class _Sampler(object):
    def __init__(self, arr):
        self.arr = copy.deepcopy(arr)

    def next(self):
        raise NotImplementedError()


class shuffle_sampler(_Sampler):
    def __init__(self, arr):
        super().__init__(arr)
        np.random.shuffle(self.arr)
        self._idx = 0
        self._max_idx = len(self.arr)

    def next(self):
        if self._idx >= self._max_idx:
            np.random.shuffle(self.arr)
            self._idx = 0
        v = self.arr[self._idx]
        self._idx += 1
        return v


class random_sampler(_Sampler):
    def next(self):
        # np.random.randint(0, int(1 / slim_ratios[0]))
        v = np.random.choice(self.arr)  # single value. If multiple value, note the replace param.
        return v


class constant_sampler(_Sampler):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def next(self):
        return self.value
