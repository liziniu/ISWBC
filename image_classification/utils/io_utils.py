import os
import shutil

from utils.logger import Logger

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def configure_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    print("Logging into: %s" % log_dir)
    logger = Logger(log_dir)
    return logger


def save_code(save_dir):
    project_dir = PROJECT_DIR
    shutil.copytree(project_dir, save_dir + '/code',
                    ignore=shutil.ignore_patterns('log*', 'result*', 'model*', 'dataset*', 'checkpoint*',
                                                  '.git', '*.pyc', '.idea', '.DS_Store'))


if __name__ == "__main__":
    print(PROJECT_DIR)
