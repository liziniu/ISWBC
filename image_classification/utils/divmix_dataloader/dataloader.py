from typing import List

from torch.utils.data import DataLoader
from torchvision import transforms

from .cifar import CifarDataset
from .dataset import ExDataset, NoisySemiSupDataset, FixMatchDataset, DivideMixDataset, CatExDataset
from .digit import CatDigitsDataset
from .domainnet import DomainNetDataset


def get_transforms(dataset):
    from .digit import DigitsDataset
    if dataset in ['cifar10', 'Cifar10+IN32']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # FIXME the norm is not used in CL. So remove.
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # FIXME the norm is not used in CL. So remove.
        ])
    elif dataset in ['CelebA+UTKFace+TIN']:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset.lower() in ['cifar100', 'cifar100c20']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),  # FIXME the norm is not used in CL. So remove.
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),  # FIXME the norm is not used in CL. So remove.
        ])
    elif dataset in DigitsDataset.all_domains + ['MNIST-F'] or dataset == 'Digits':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        transform_test = transforms.ToTensor()
    elif dataset in DomainNetDataset.all_domains or dataset == 'DomainNet':
        transform_train = transforms.Compose([
            transforms.Resize([244, 244]),
            # transforms.Resize([224, 224]),
            # transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize([256, 256]),
            transforms.Resize([244, 244]),
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"dataset: {dataset}")
    return transform_train, transform_test


def get_dataset(name, id_domains, ood_domains, ext_domains, target_type='Smiling'):
    from .digit import DigitsDataset
    ood_train_dataset, ood_test_dataset = None, None
    # if self.data_name.lower() in ['cifar10']:
    #     self.dataset = self.dataset.lower()
    #     from .cifar10 import cifar_dataset
    #     self.DataClass = cifar_dataset
    # el
    if name in ['MNIST-F']:
        from .mnist import MNIST
        # self.DataClass = MNIST
        train_dataset = MNIST(split='train')
        test_dataset = MNIST(split='test')
    elif name in DigitsDataset.all_domains:
        # self.DataClass = DigitsDataset
        train_dataset = DigitsDataset(split='train', domain=name)
        test_dataset = DigitsDataset(split='test', domain=name)
        num_classes = 10
    elif name == 'Digits':
        domain_names = DigitsDataset.all_domains
        id_domains = str2domain_names(id_domains, domain_names)
        if ext_domains == 'none':
            ext_domains = []
        else:
            ext_domains = [d for d in ext_domains.split(',')]
        if ood_domains == 'id':
            ood_domains = id_domains
        else:
            ood_domains = str2domain_names(ood_domains, domain_names, id_domains)
        if any([d in ood_domains for d in id_domains]):
            assert all([d in ood_domains for d in id_domains]), "cannot make partial overlap."
            include_id_to_ood = True
        else:
            include_id_to_ood = False
        ood_domains = [d for d in ood_domains if d not in id_domains]  # remove ID domains

        # ID data will be split into client/query
        train_dataset = CatDigitsDataset(
            [DigitsDataset(split='train', domain=d) for d in id_domains])
        test_dataset = CatDigitsDataset(
            [DigitsDataset(split='test', domain=d) for d in id_domains])
        num_classes = len(train_dataset.datasets[0].classes)
        print(" ID domains: " + ", ".join(
            [f"{ds.domain}: {len(ds)}" for ds in train_dataset.datasets]))

        if len(ood_domains) > 0:
            # OoD data will be only used in unlabeled query or test.
            tr_ds = [DigitsDataset(split='train', domain=d) for d in ood_domains]
            if len(ext_domains) > 0:
                tr_ds += [
                    CifarDataset(domain=d, split='train', transform=transforms.Resize((28, 28)))
                    for d in ext_domains]
            ood_train_dataset = CatDigitsDataset(tr_ds)
            te_ds = [DigitsDataset(split='test', domain=d) for d in ood_domains]
            if len(ext_domains) > 0:
                te_ds += [
                    CifarDataset(domain=d, split='test', transform=transforms.Resize((28, 28)))
                    for d in ext_domains]
            ood_test_dataset = CatDigitsDataset(te_ds)

            print(" OoD domains: " + ", ".join(
                [f"{ds.domain}: {len(ds)}" for ds in ood_train_dataset.datasets]))
    elif name == 'DomainNet':
        domain_names = DomainNetDataset.all_domains
        id_domains = str2domain_names(id_domains, domain_names)
        if ext_domains == 'none':
            ext_domains = []
        else:
            ext_domains = [d for d in ext_domains.split(',')]
        if ood_domains == 'id':
            ood_domains = id_domains
        else:
            ood_domains = str2domain_names(ood_domains, domain_names, id_domains)
        if any([d in ood_domains for d in id_domains]):
            assert all([d in ood_domains for d in id_domains]), "cannot make partial overlap."
            include_id_to_ood = True
        else:
            include_id_to_ood = False
        ood_domains = [d for d in ood_domains if d not in id_domains]  # remove ID domains
        # ID data will be split into client/query
        train_dataset = CatExDataset(
            [DomainNetDataset(split='train', domain=d) for d in id_domains])
        test_dataset = CatExDataset(
            [DomainNetDataset(split='test', domain=d) for d in id_domains])
        num_classes = len(train_dataset.datasets[0].classes)
        print(" ID domains: " + ", ".join(
            [f"{ds.domain}: {len(ds)}" for ds in train_dataset.datasets]))
        # import pdb; pdb.set_trace()
        if len(ood_domains) > 0:
            # OoD data will be only used in unlabeled query or test.
            tr_ds = [DomainNetDataset(split='train', domain=d) for d in ood_domains]
            if len(ext_domains) > 0:
                assert ext_domains == ['CelebA']
                from ..dataset import CelebA
                from utils.config import data_root
                # raise NotImplementedError()
                # tr_ds += [CifarDataset(domain=d, split='train', transform=transforms.Resize((28, 28)))
                #           for d in ext_domains]
                tr_ds += [CelebA(data_root, split='train', target_type=None,
                                 transform=None, percent=0.05)]
            ood_train_dataset = CatExDataset(tr_ds)
            te_ds = [DomainNetDataset(split='test', domain=d) for d in ood_domains]
            if len(ext_domains) > 0:
                # raise NotImplementedError()
                tr_ds += [CelebA(data_root, split='test', target_type=None,
                                 transform=None)]
            ood_test_dataset = CatExDataset(te_ds)
            print(" OoD domains: " + ", ".join(
                [f"{ds.domain}: {len(ds)}" for ds in ood_train_dataset.datasets]))
    elif name == 'Cifar100C20':
        from .cifar100 import Cifar100C20Dataset
        domain_names = Cifar100C20Dataset.all_domains
        id_domains = str2domain_names(id_domains, domain_names)
        if ext_domains == 'none':
            ext_domains = []
        else:
            ext_domains = [d for d in ext_domains.split(',')]
        if ood_domains == 'id':
            ood_domains = id_domains
        else:
            ood_domains = str2domain_names(ood_domains, domain_names, id_domains)
        if any([d in ood_domains for d in id_domains]):
            assert all([d in ood_domains for d in id_domains]), "cannot make partial overlap."
            include_id_to_ood = True
        else:
            include_id_to_ood = False
        ood_domains = [d for d in ood_domains if d not in id_domains]  # remove ID domains

        # ID data will be split into client/query
        train_dataset = CatExDataset(
            [Cifar100C20Dataset(split='train', domain=d) for d in id_domains])
        test_dataset = CatExDataset(
            [Cifar100C20Dataset(split='test', domain=d) for d in id_domains])
        num_classes = len(train_dataset.datasets[0].classes)
        print(" ID domains: " + ", ".join(
            [f"{ds.domain}: {len(ds)}" for ds in train_dataset.datasets]))

        if len(ood_domains) > 0:
            # OoD data will be only used in unlabeled query or test.
            tr_ds = [Cifar100C20Dataset(split='train', domain=d) for d in ood_domains]
            if len(ext_domains) > 0:
                raise NotImplementedError()
                # tr_ds += [CifarDataset(domain=d, split='train', transform=transforms.Resize((28, 28)))
                #           for d in ext_domains]
            ood_train_dataset = CatExDataset(tr_ds)
            te_ds = [Cifar100C20Dataset(split='test', domain=d) for d in ood_domains]
            if len(ext_domains) > 0:
                raise NotImplementedError()
                # te_ds += [CifarDataset(domain=d, split='test', transform=transforms.Resize((28,28)))
                #           for d in ext_domains]
            ood_test_dataset = CatExDataset(te_ds)

            print(" OoD domains: " + ", ".join(
                [f"{ds.domain}: {len(ds)}" for ds in ood_train_dataset.datasets]))
    elif name == 'Cifar10+IN32':
        from .cifar10_in32 import Cifar10Dataset
        from .imagenet import ImageNetDS
        domain_names = ['Cifar10', 'IN32']
        assert id_domains == '0', f'invalid id_domains: {id_domains}'
        id_domains = str2domain_names(id_domains, domain_names)

        # assert ood_domains == 'ex_id', f'invalid ood_domains: {ood_domains}'
        # id_domains = str2domain_names(id_domains, domain_names)
        if ext_domains == 'none':
            ext_domains = []
        else:
            ext_domains = [d for d in ext_domains.split(',')]
        if ood_domains == 'id':
            ood_domains = id_domains
        else:
            ood_domains = str2domain_names(ood_domains, domain_names, id_domains)
        if any([d in ood_domains for d in id_domains]):
            assert all([d in ood_domains for d in id_domains]), "cannot make partial overlap."
            include_id_to_ood = True
        else:
            include_id_to_ood = False
        ood_domains = [d for d in ood_domains if d not in id_domains]  # remove ID domains

        # ID data will be split into client/query
        train_dataset = Cifar10Dataset(split='train')
        test_dataset = Cifar10Dataset(split='test')
        num_classes = len(train_dataset.classes)
        print(" ID domains: " + ", ".join(
            [f"{ds.domain}: {len(ds)}" for ds in [train_dataset]]))

        if len(ood_domains) > 0:
            assert len(ood_domains) == 1
            assert ood_domains[0] == 'IN32'
            # OoD data will be only used in unlabeled query or test.
            tr_ds = ImageNetDS(img_size=32, train=True, percent=0.08, unlabel_targets=True)
            tr_ds.domain_id = domain_names.index(ood_domains[0])
            if len(ext_domains) > 0:
                raise NotImplementedError()
            ood_train_dataset = tr_ds
            # tr_ds = ImageNetDS(img_size=32, train=False)
            if len(ext_domains) > 0:
                raise NotImplementedError()
            # ood_test_dataset = CatExDataset(te_ds)
            ood_test_dataset = None

            print(" OoD domains: " + ", ".join(
                [f"{ds.domain}: {len(ds)}" for ds in [ood_train_dataset]]))
    elif name == 'CelebA+UTKFace+TIN':
        # from . import Cifar10Dataset
        from .imagenet import TinyImageNet
        from ..dataset import UTKFace, CelebA
        from utils.config import data_root
        assert id_domains == '0', f'invalid id_domains: {id_domains}'
        domain_names = ['CelebA', 'UTKFace', 'TIN']
        id_domains = str2domain_names(id_domains, domain_names)

        # domain_names = Cifar100C20Dataset.all_domains
        # assert ood_domains == 'ex_id', f'invalid ood_domains: {ood_domains}'
        # id_domains = str2domain_names(id_domains, domain_names)
        if ext_domains == 'none':
            ext_domains = []
        else:
            ext_domains = [d for d in ext_domains.split(',')]
        if ood_domains == 'id':
            ood_domains = id_domains
        else:
            ood_domains = str2domain_names(ood_domains, domain_names, id_domains)
        if any([d in ood_domains for d in id_domains]):
            assert all([d in ood_domains for d in id_domains]), "cannot make partial overlap."
            include_id_to_ood = True
        else:
            include_id_to_ood = False
        ood_domains = [d for d in ood_domains if d not in id_domains]  # remove ID domains

        # ID data will be split into client/query
        assert id_domains == ['CelebA'], f'invalid id_domains: {id_domains}'
        train_dataset = CelebA(data_root, split='train', target_type=target_type,
                               transform=transforms.Resize((64, 64)), percent=0.1)
        test_dataset = CelebA(data_root, split='test', target_type=target_type,
                              transform=transforms.Resize((64, 64)))
        num_classes = len(train_dataset.classes)
        print(" ID domains: " + ", ".join(
            [f"{ds.domain}: {len(ds)}" for ds in [train_dataset]]))

        if len(ood_domains) > 0:
            all_ds = []
            # OoD data will be only used in unlabeled query or test.
            if 'UTKFace' in ood_domains:
                tr_ds0 = UTKFace(data_root, target_type='young', return_attr=False,
                                 transform=transforms.Resize((64, 64)))
                tr_ds0.domain_id = domain_names.index('UTKFace')
                all_ds.append(tr_ds0)
            if 'TIN' in ood_domains:
                tr_ds1 = TinyImageNet(split='train')
                tr_ds1.domain_id = domain_names.index('TIN')
                all_ds.append(tr_ds1)
            if len(ext_domains) > 0:
                raise NotImplementedError()
            ood_train_dataset = CatExDataset(all_ds)
            # tr_ds = ImageNetDS(img_size=32, train=False)
            # ood_test_dataset = CatExDataset(te_ds)
            ood_test_dataset = None

            print(" OoD domains: " + ", ".join(
                [f"{ds.domain}: {len(ds)}" for ds in ood_train_dataset.datasets]))
    else:
        raise NotImplementedError(f"dataset: {name}")
    return train_dataset, test_dataset, ood_train_dataset, ood_test_dataset, num_classes, include_id_to_ood


def str2domain_names(s: str, domains: List[str], id_domains: List[str] = None):
    if s == 'all':
        names = domains
    elif s == 'ex_id':  # exclude ID
        assert id_domains is not None and len(id_domains) > 0, "Require ID domains for ex_id"
        names = [d for d in domains if d not in id_domains]
    else:
        names = [domains[int(d)] for d in s.split(',')]
        # for i in names:
        #     assert i < len(domains), f"Invalid domain id: {i}"
    return names


class DataloaderManager:
    """Quickly access different dataloaders for different purposes."""

    def __init__(self, dataset, batch_size=128, num_workers=4,
                 r=0., noise_mode='asym',
                 train_mode='divmix',
                 new_noise_file=True, noise_label=None,
                 client_ratio=0.,
                 id_domains='0', ood_domains='1', ext_domains='none',
                 target_type='Smiling',
                 ):
        """If `noise_label` is None, will try to generate noisy labels or load from noise_file.

        r, noise_mode and num_labeled are only used when noise_label is not provided on `run`.
        """
        self.data_name = dataset
        self.train_mode = train_mode
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_label = noise_label
        self.sample_weights = None
        self.noise_file = f'noise_files/{dataset.lower()}/{r}_{noise_mode}.json'
        self.new_noise_file = new_noise_file
        # print(f"will use noise file: {noise_file}")
        self.client_ratio = client_ratio

        self.train_dataset, self.test_dataset, self.ood_train_dataset, self.ood_test_dataset, self.num_classes, self.include_id_to_ood = get_dataset(
            dataset, id_domains, ood_domains, ext_domains, target_type=target_type)

        assert isinstance(self.train_dataset, ExDataset), f"Invalid dataset " \
                                                          f"type: {type(self.train_dataset)}"
        assert isinstance(self.test_dataset, ExDataset), f"Invalid dataset " \
                                                         f"type: {type(self.test_dataset)}"
        self.transform_train, self.transform_test = get_transforms(self.data_name)

    def run(self, mode,  # pred=np.empty(0), prob=np.empty(0),
            client_aug=False):
        dataset = self.test_dataset if mode == 'test' else self.train_dataset
        ood_dataset = None
        if self.ood_test_dataset is not None and mode == 'test':
            ood_dataset = self.ood_test_dataset
        if self.ood_train_dataset is not None and mode != 'test':
            ood_dataset = self.ood_train_dataset
        # dataset = self.DataClass(split='test' if mode == 'test' else 'train')
        if self.train_mode in ['sup', 'vat', 'shot', 'kd']:
            DSSD = NoisySemiSupDataset
        elif self.train_mode in ['divmix']:
            DSSD = DivideMixDataset
        elif self.train_mode in ['fixmatch', 'flexmatch']:
            DSSD = FixMatchDataset
        else:
            raise NotImplementedError(f"self.train_mode: {self.train_mode}")
        if mode == 'warmup':
            all_dataset = DSSD(
                dataset, mode="noise_labeled", transform=self.transform_train,
                client_ratio=self.client_ratio, noise_label=self.noise_label, sample_weights=self.sample_weights,
                noise_mode=self.noise_mode, noise_ratio=self.r,
                noise_file=self.noise_file, new_noise_file=self.new_noise_file,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = DSSD(
                dataset, mode="labeled", transform=self.transform_train, noise_mode=self.noise_mode,
                noise_ratio=self.r, noise_file=self.noise_file, new_noise_file=self.new_noise_file,
                # pred=pred, probability=prob,
                noise_label=self.noise_label, sample_weights=self.sample_weights,
                client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            if len(labeled_dataset) > 0:
                labeled_trainloader = DataLoader(
                    dataset=labeled_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers)
            else:
                labeled_trainloader = None

            unlabeled_dataset = DSSD(
                dataset, mode="unlabeled", transform=self.transform_train,
                noise_mode=self.noise_mode, noise_ratio=self.r,
                noise_file=self.noise_file,  # pred=pred,
                new_noise_file=self.new_noise_file,
                noise_label=self.noise_label, sample_weights=self.sample_weights,
                client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            if len(unlabeled_dataset) > 0:
                unlabeled_trainloader = DataLoader(
                    dataset=unlabeled_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers)
            else:
                unlabeled_trainloader = None

            run_ood_dataset = DSSD(
                dataset, mode="ood", transform=self.transform_train,
                noise_mode=self.noise_mode, noise_ratio=self.r,
                noise_file=self.noise_file,  # pred=pred,
                new_noise_file=self.new_noise_file,
                noise_label=self.noise_label, sample_weights=self.sample_weights,
                client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            if len(run_ood_dataset) > 0:
                run_ood_trainloader = DataLoader(
                    dataset=run_ood_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers)
            else:
                run_ood_trainloader = None
            return labeled_trainloader, unlabeled_trainloader, run_ood_trainloader

        elif mode == 'eval_train':
            eval_dataset = DSSD(
                dataset, mode='noise_labeled', transform=self.transform_test,
                noise_mode=self.noise_mode, noise_ratio=self.r,
                noise_file=self.noise_file, new_noise_file=self.new_noise_file,
                noise_label=self.noise_label, sample_weights=self.sample_weights, client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = DSSD(
                dataset, mode='test', transform=self.transform_test,
                # ood_dataset=ood_dataset,  # FIXME should include this depending on arg.
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,  # * 2,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode in ['client']:
            client_dataset = DSSD(
                dataset, mode=mode, transform=self.transform_train if client_aug else self.transform_test,
                client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            print(f"Client data size: {len(client_dataset)}")
            client_loader = DataLoader(
                dataset=client_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return client_loader

        elif mode in ['query']:
            query_dataset = DSSD(
                dataset, mode=mode, transform=self.transform_test,
                client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            print(f"Query data size: {len(query_dataset)}")
            client_loader = DataLoader(
                dataset=query_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return client_loader

        elif mode in ['client_train']:  # for training, we will augment data and shuffle.
            client_dataset = DSSD(
                dataset, mode='client', transform=self.transform_train,
                client_ratio=self.client_ratio,
                ood_dataset=ood_dataset, include_id_to_ood=self.include_id_to_ood,
            )
            print(f"Client data size: {len(client_dataset)}")
            client_loader = DataLoader(
                dataset=client_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return client_loader
