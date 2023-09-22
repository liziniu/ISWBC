"""Configuration file for defining paths to data."""
import os


def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)


hostname = os.uname()[1]  # type: str
# Update your paths here.
# CHECKPOINT_ROOT = './checkpoint'
# if hostname.startswith('illidan') and int(hostname.split('-')[-1]) >= 8:
#     data_root = '/localscratch2/jyhong/'
# elif hostname.startswith('illidan'):
#     data_root = '/media/Research/jyhong/data'
# elif hostname.startswith('ip-'):  # aws
#     data_root = os.path.expanduser('~/data')
# else:
#     data_root = './data'
CHECKPOINT_ROOT = '/home/znli/project/PrivateSampling/checkpoint'  # '/ssddata1/data/zeyuqin/PrivateSampling/checkpoint'
data_root = '/home/znli/project/datasets'  # '/ssddata1/data/zeyuqin/ORCA/datasets'
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)

DATA_PATHS = {
    "DomainNet": data_root + '/DomainNET',  # data_root + "/domainnet",
    "DomainNetPathList": data_root + "/domainnet10",
    # store the path list file from FedBN
    "Cifar10": data_root + "/cifar10",
    "Cifar100": data_root + "/cifar100",
    "Adult": os.path.join(data_root, 'adult'),
    "MNIST-F": data_root + "/MNIST",
    "MNIST": data_root + "/MNIST",
    "WikiText2": data_root,
    "TinyImageNet": data_root,
    "ImageNetDS": data_root,
}

# if hostname.startswith('illidan') and int(hostname.split('-')[-1]) < 8:
#     # personal config
#     home_path = os.path.expanduser('~/')
#     DATA_PATHS = {
#         **DATA_PATHS,
#         "Digits": home_path + "projects/FedBN/data",
#         "DomainNetPathList": home_path + "projects/FedBN/data/DomainNet",  # store the path list file from FedBN
#         "OfficePathList": home_path + "projects/FedBN/data/office_caltech_10",
#     }
# else:
DATA_PATHS = {
    **DATA_PATHS,
    "Digits": data_root + "/Digits",
    "DomainNetPathList": data_root + "/domainnet10/DomainNet"
    # "DomainNetPathList": data_root + "/domainnet/domainnet10/DomainNet",  # store the path list file from FedBN
    # "OfficePathList": home_path + "projects/FedBN/data/office_caltech_10",
}
