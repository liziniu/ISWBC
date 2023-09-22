import os
import time
import sys
from subprocess import check_output, CalledProcessError
import numpy as np
import tensorflow.compat.v2 as tf
import shutil
from logger import Logger

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ))


def set_global_seed(seed=2019):
    import tensorflow.compat.v2 as tf
    import random
    assert seed > 0
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def configure_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    print("Logging into: %s" % log_dir)
    logger = Logger(log_dir)
    return logger


def save_code(save_dir):
    project_dir = PROJECT_DIR
    shutil.copytree(project_dir, save_dir + '/code',
                    ignore=shutil.ignore_patterns('log*', 'result*', 'gail-experts', 'dataset*', 'checkpoint*',
                                                  '.git', '*.pyc', '.idea', '.DS_Store'))


if __name__ == "__main__":
    print(PROJECT_DIR)
