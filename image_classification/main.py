from __future__ import print_function

import argparse
import copy
import os
import time

# import umap
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import io_utils
from utils.divmix_dataloader.dataloader import get_dataset
from utils.divmix_dataloader.dataset import FixMatchDataset
from utils.utils import set_seed


# from otdd.pytorch.distance import DatasetDistance, FeatureCost

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--train_mode', type=str, default='fixmatch')
    parser.add_argument('--algo', type=str, default='bc', choices=['bc', 'nbcu', 'iswbc'])
    # domain: feature covariate shift
    parser.add_argument('--id_domains', type=str, default='0', choices=['0', '1', '2', '3', '4', '5'])
    parser.add_argument('--ood_domains', type=str, default='ex_id', help='`all` for all domains, '
                                                                         '`id` for the same as `id_domains`, '
                                                                         '`0,1` for 0 and 1 domains')
    parser.add_argument('--ext_domains', type=str, default='none', help='extended OoD domains. '
                                                                        'Specify by name, e.g.,'
                                                                        ' `Cifar10,Cifar100`')

    parser.add_argument('--client_ratio', type=float, default=1.0)  # 0.5 half target dataset
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr_clas', type=float, default=0.01)
    parser.add_argument('--lr_disc', type=float, default=0.01)
    parser.add_argument('--weight_decay_disc', type=float, default=5e-4)
    parser.add_argument('--weight_decay_clas', type=float, default=5e-4)
    parser.add_argument('--epochs_disc', type=int, default=100)
    parser.add_argument('--epochs_clas', type=int, default=100)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--optimizer_disc', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--optimizer_clas', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--num_layers_disc', type=int, default=2)
    parser.add_argument('--num_layers_clas', type=int, default=1)
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=None)
    parser.add_argument('--filename', type=str, default='exp')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')
    parser.add_argument('--embeddings', default=False,
                        action="store_true",
                        help='whether to use embeddings')
    parser.add_argument('--path_hh', type=str,
                        # default='/ssddata1/data/zeyuqin/PrivateSampling/tsne_results'
                        default='/home/znli/project/PrivateSampling/tsne_results'
                        )
    parser.add_argument('--vis', type=str, default='tsne')

    return parser.parse_args()


class Discriminator(nn.Module):
    """
    Discriminator that is trained for recognizing in/ood samples
    """

    def __init__(self, num_layers):
        super().__init__()
        self.label_net = nn.Sequential(
            nn.Linear(10, 512),
            # nn.ReLU(inplace=True)
        )

        assert num_layers >= 1
        head_net = []
        for i in range(num_layers - 1):
            head_net.append(nn.Linear(1024, 1024))
            head_net.append(nn.ReLU(inplace=True))
        head_net.append(nn.Linear(1024, 1))
        self.head_net = nn.Sequential(*head_net)

        for param in self.label_net.parameters():
            param.requires_grad_ = False

    def __call__(self, features, labels):
        labels = F.one_hot(labels, num_classes=10).to(torch.float32)
        # import pdb; pdb.set_trace()
        # print(labels.shape)

        label_features = self.label_net(labels)

        torch_version_1 = int(torch.__version__.split('.')[0])
        torch_version_2 = int(torch.__version__.split('.')[1])
        if torch_version_1 == 1 and torch_version_2 <= 7:  # less than '1.7.0'
            feature = torch.cat([features, label_features], dim=-1)
        else:
            feature = torch.concat([features, label_features], dim=-1)
        logit = self.head_net(feature)
        return logit


class Classifier(nn.Module):
    """
    Classifier for image classification
    """

    def __init__(self, hidden_layer_size, num_class, num_layers):
        super().__init__()
        assert num_layers >= 1
        head_net = []

        for i in range(num_layers - 1):
            head_net.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            head_net.append(nn.ReLU(inplace=True))
        head_net.append(nn.Linear(hidden_layer_size, num_class))
        self.head_net = nn.Sequential(*head_net)

    def __call__(self, features):
        logit = self.head_net(features)
        return logit


def test(classifier, query_loader_test, device):
    classifier.eval()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    wrong_list = torch.tensor([]).long().to(device)
    for data in query_loader_test:
        features, targets = data
        features, targets = features.to(device), targets.to(device)
        outputs = classifier(features)
        loss = F.cross_entropy(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        idx += 1
        wrong_list = torch.cat((wrong_list, targets[~predicted.eq(targets)]))

    return train_loss / idx, correct / total, wrong_list


def create_datasets(args, device, logger):
    # create model
    if args.model == 'resnet50':
        model = models.__dict__['resnet50'](pretrained=True).to(device)

    elif args.model == 'resnet18':
        model = models.__dict__['resnet18'](pretrained=True).to(device)

    elif args.model == 'wide_resnet50':
        model = models.__dict__['wide_resnet50_2'](pretrained=True).to(device)

    elif args.model == 'vgg16_bn':
        model = models.__dict__['vgg16_bn'](pretrained=True).to(device)

    elif args.model == 'efficientnet_b6':
        model = models.__dict__['efficientnet_b6'](pretrained=True).to(device)

    elif args.model == 'convnext_small':
        model = models.__dict__['convnext_small'](pretrained=True).to(device)

    elif args.model == 'convnext_tiny':
        model = models.__dict__['convnext_tiny'](pretrained=True).to(device)

    elif args.model == 'convnext_base':
        model = models.__dict__['convnext_base'](pretrained=True).to(device)
        model = copy.deepcopy(model)
        model.classifier[-1] = nn.Identity()

    elif args.model == 'swin_b':
        model = models.__dict__['swin_b'](pretrained=True).to(device)

    elif args.model == 'swin_t':
        model = models.__dict__['swin_t'](pretrained=True).to(device)

    elif args.model == 'instagram_resnext101_32x8d':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)

    elif args.model == 'bit_m_resnet50':
        import big_transfer.bit_pytorch.models as bit_models
        model = bit_models.KNOWN_MODELS['BiT-M-R50x1'](zero_head=True)
        model.load_from(np.load('BiT-M-R50x1.npz'))
        model = model.to(device)

    elif args.model == 'hug_sw_b':
        raise NotImplementedError
        # arch_name = "/ssddata1/data/zeyuqin/ORCA/pretrained_models/swin-base-patch4-window7-224-in22k"
        # # arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
        # embed_dim = 128
        # output_dim = 1024
        # img_size = 224
        # patch_size = 4
        # modelclass = SwinForImageClassification
        #
        # model = modelclass.from_pretrained(arch_name)
        # model.config.image_size = img_size
        # model = modelclass.from_pretrained(arch_name, config=model.config)
        # model.pooler = nn.AdaptiveAvgPool1d(1)
        # model.classifier = nn.Identity()
        # model = copy.deepcopy(model)
        # model = model.to(device)

    elif args.model == 'clip':
        # load_path = '/ssddata1/data/zeyuqin/pretrain_models/checkpoints'
        load_path = '/home/znli/project/pretrain_models/checkpoints'
        model, preprocess = clip.load('ViT-B/32', device, download_root=load_path)
        # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    model = copy.deepcopy(model)
    if 'res' in args.model:
        model.fc = nn.Identity()
    elif 'conv' in args.model:
        model.classifier[-1] = nn.Identity()
    elif 'swin' in args.model:
        model.head = nn.Identity()
    elif 'vgg' in args.model:
        model.classifier[6] = nn.Identity()

    # import pdb; pdb.set_trace()

    model.eval()

    # import pdb; pdb.set_trace()

    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        # transforms.Resize([244, 244]),
        transforms.Resize([224, 224]),
        # transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        normalize_transform,
    ])

    transform_test = transforms.Compose([
        # transforms.Resize([256, 256]),
        # transforms.Resize([244, 244]),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize_transform,
    ])

    if args.model == 'clip':
        transform_test = preprocess
        transform_train = preprocess

    train_dataset, test_dataset, ood_train_dataset, ood_test_dataset, num_classes, include_id_to_ood = get_dataset(
        'DomainNet', args.id_domains, args.ood_domains, args.ext_domains, target_type='Smiling')

    DSSD = FixMatchDataset

    transform_train = transform_test
    # client_aug = False
    client_dataset = DSSD(
        train_dataset, mode='client', transform=transform_train,
        client_ratio=args.client_ratio,
        ood_dataset=ood_train_dataset, include_id_to_ood=include_id_to_ood,
    )
    logger.info(f"Client data size: {len(client_dataset)}")
    client_loader = DataLoader(
        dataset=client_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    # client_aug = False
    client_dataset_test = DSSD(
        test_dataset, mode='test', transform=transform_test,
        # client_ratio=args.client_ratio,
        # ood_dataset=ood_train_dataset, include_id_to_ood=include_id_to_ood,
    )
    logger.info(f"Client testing data size: {len(client_dataset_test)}")
    client_loader_test = DataLoader(
        dataset=client_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    query_dataset = DSSD(
        train_dataset, mode='query', transform=transform_train,
        client_ratio=args.client_ratio,
        ood_dataset=ood_train_dataset, include_id_to_ood=include_id_to_ood,
    )
    logger.info(f"Query data size: {len(query_dataset)}")
    query_loader = DataLoader(
        dataset=query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    query_dataset_test = DSSD(
        ood_test_dataset, mode='test', transform=transform_test,
        # client_ratio=args.client_ratio,
        # ood_dataset=ood_train_dataset, include_id_to_ood=include_id_to_ood,
    )
    logger.info(f"Query testing data size: {len(query_dataset_test)}")
    query_loader_test = DataLoader(
        dataset=query_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    # for data in query_loader:

    # print(data['target'].shape)

    # import pdb; pdb.set_trace()

    iii = 0
    for data in client_loader:
        images, targets = data['image'].to(device), data['target'].to(device)

        if iii == 0:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    id_features = model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()
                    # import pdb; pdb.set_trace()
                else:
                    id_features = model(images).logits.detach().cpu().numpy()
            elif 'clip' in args.model:
                id_features = model.encode_image(images).detach().cpu().numpy()
            else:
                id_features = model(images).detach().cpu().numpy()
            # new_feature = model.compute_features(images).detach().cpu().numpy()
            id_labels = targets.detach().cpu().numpy()
        else:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    id_features = np.concatenate(
                        (id_features, model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()))
                else:
                    id_features = np.concatenate((id_features, model(images).logits.detach().cpu().numpy()))
            elif 'clip' in args.model:
                id_features = np.concatenate((id_features, model.encode_image(images).detach().cpu().numpy()))
            else:
                id_features = np.concatenate((id_features, model(images).detach().cpu().numpy()))
            # new_feature = np.concatenate((new_feature,model.compute_features(images).detach().cpu().numpy()))
            id_labels = np.concatenate((id_labels, targets.detach().cpu().numpy()))
        # print(id_features.shape)
        # print(id_labels.shape)
        iii = iii + 1

    iii = 0
    for data in client_loader_test:
        images, targets = data['image'].to(device), data['target'].to(device)

        if iii == 0:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    id_features_test = model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()
                    # import pdb; pdb.set_trace()
                else:
                    id_features_test = model(images).logits.detach().cpu().numpy()
            elif 'clip' in args.model:
                id_features_test = model.encode_image(images).detach().cpu().numpy()
            else:
                id_features_test = model(images).detach().cpu().numpy()
            # new_feature = model.compute_features(images).detach().cpu().numpy()
            id_labels_test = targets.detach().cpu().numpy()
        else:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    id_features_test = np.concatenate(
                        (id_features_test, model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()))
                else:
                    id_features_test = np.concatenate((id_features_test, model(images).logits.detach().cpu().numpy()))
            elif 'clip' in args.model:
                id_features_test = np.concatenate((id_features_test, model.encode_image(images).detach().cpu().numpy()))
            else:
                id_features_test = np.concatenate((id_features_test, model(images).detach().cpu().numpy()))
            # new_feature = np.concatenate((new_feature,model.compute_features(images).detach().cpu().numpy()))
            id_labels_test = np.concatenate((id_labels_test, targets.detach().cpu().numpy()))
        # print(id_features.shape)
        # print(id_labels.shape)
        iii = iii + 1

    id_features = torch.from_numpy(id_features).float().to('cpu')
    id_labels = torch.from_numpy(id_labels).long().to('cpu')

    id_dataset = torch.utils.data.TensorDataset(id_features, id_labels)

    id_loader = DataLoader(
        dataset=id_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    id_features_test = torch.from_numpy(id_features_test).float().to('cpu')
    id_labels_test = torch.from_numpy(id_labels_test).long().to('cpu')

    id_dataset_test = torch.utils.data.TensorDataset(id_features_test, id_labels_test)

    id_loader_test = DataLoader(
        dataset=id_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    '''
    id_mask = torch.ones(10)
    ii = 0
    for data in query_dataset:
        images, targets = data['image'], data['target']
        if id_mask.sum() == 0:
            break
        id_mask[targets] = 0
        img = images.numpy().transpose(1,2,0)*255.0
        img = np.clip(img.astype('uint8'), 0, 255)
        out_file = '/ssddata1/data/zeyuqin/PrivateSampling/tsne_results/' +str(targets) +'.png'
        matplotlib.image.imsave(out_file, img)

    draw the pictures of domainnet dataset
    '''

    iii = 0
    for data in query_loader:
        images, targets, domains = data['image'].to(device), data['target'].to(device), data['domain'].to(device)

        if iii == 0:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    new_feature = model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()
                    # import pdb; pdb.set_trace()
                else:
                    new_feature = model(images).logits.detach().cpu().numpy()
            elif 'clip' in args.model:
                new_feature = model.encode_image(images).detach().cpu().numpy()
            else:
                new_feature = model(images).detach().cpu().numpy()
            # new_feature = model.compute_features(images).detach().cpu().numpy()
            new_labels = targets.detach().cpu().numpy()
            new_domains = domains.detach().cpu().numpy()
        else:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    new_feature = np.concatenate(
                        (new_feature, model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()))
                else:
                    new_feature = np.concatenate((new_feature, model(images).logits.detach().cpu().numpy()))
            elif 'clip' in args.model:
                new_feature = np.concatenate((new_feature, model.encode_image(images).detach().cpu().numpy()))
            else:
                new_feature = np.concatenate((new_feature, model(images).detach().cpu().numpy()))
            # new_feature = np.concatenate((new_feature,model.compute_features(images).detach().cpu().numpy()))
            new_labels = np.concatenate((new_labels, targets.detach().cpu().numpy()))
            new_domains = np.concatenate((new_domains, domains.detach().cpu().numpy()))
        # print(new_feature.shape)
        # print(new_labels.shape)
        # print(new_domains.shape)
        iii = iii + 1

    iii = 0
    for data in query_loader_test:
        images, targets, domains = data['image'].to(device), data['target'].to(device), data['domain'].to(device)

        if iii == 0:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    new_feature_test = model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()
                    # import pdb; pdb.set_trace()
                else:
                    new_feature_test = model(images).logits.detach().cpu().numpy()
            elif 'clip' in args.model:
                new_feature_test = model.encode_image(images).detach().cpu().numpy()
            else:
                new_feature_test = model(images).detach().cpu().numpy()
            # new_feature = model.compute_features(images).detach().cpu().numpy()
            new_labels_test = targets.detach().cpu().numpy()
            new_domains_test = domains.detach().cpu().numpy()
        else:
            if 'hug_sw_b' in args.model:
                if args.embeddings:
                    new_feature_test = np.concatenate(
                        (new_feature_test, model.swin.embeddings(images)[0].mean(1).detach().cpu().numpy()))
                else:
                    new_feature_test = np.concatenate((new_feature_test, model(images).logits.detach().cpu().numpy()))
            elif 'clip' in args.model:
                new_feature_test = np.concatenate((new_feature_test, model.encode_image(images).detach().cpu().numpy()))
            else:
                new_feature_test = np.concatenate((new_feature_test, model(images).detach().cpu().numpy()))
            # new_feature = np.concatenate((new_feature,model.compute_features(images).detach().cpu().numpy()))
            new_labels_test = np.concatenate((new_labels_test, targets.detach().cpu().numpy()))
            new_domains_test = np.concatenate((new_domains_test, domains.detach().cpu().numpy()))
        # print(new_feature.shape)
        # print(new_labels.shape)
        # print(new_domains.shape)
        iii = iii + 1

    new_feature = torch.from_numpy(new_feature).float().to('cpu')
    new_labels = torch.from_numpy(new_labels).long().to('cpu')

    new_feature_test = torch.from_numpy(new_feature_test).float().to('cpu')
    new_labels_test = torch.from_numpy(new_labels_test).long().to('cpu')

    ood_dataset = torch.utils.data.TensorDataset(new_feature, new_labels)

    ood_dataset_test = torch.utils.data.TensorDataset(new_feature_test, new_labels_test)

    ood_loader = DataLoader(
        dataset=ood_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    ood_loader_test = DataLoader(
        dataset=ood_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    union_features = torch.cat((id_features, new_feature))
    union_labels = torch.cat((id_labels, new_labels))

    union_dataset = torch.utils.data.TensorDataset(union_features, union_labels)

    union_loader = DataLoader(
        dataset=union_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    return {
        "id_features": id_features,
        "id_loader": id_loader,
        "id_loader_test": id_loader_test,
        "ood_loader": ood_loader,
        "ood_loader_test": ood_loader_test,
        "union_loader": union_loader
    }


def main(args=parse_args()):
    torch.hub.set_dir('/home/znli/project/pretrain_models/')

    save_dir = os.path.join(
        'log',
        '{}-{}-{}-{}'.format(args.algo, args.id_domains, args.seed, time.strftime('%Y-%m-%d-%H-%M-%S'))
    )
    logger = io_utils.configure_logger(save_dir)
    io_utils.save_code(save_dir)
    yaml.safe_dump(args.__dict__, open(os.path.join(save_dir, 'config.yml'), 'w'), default_flow_style=False)
    writer = SummaryWriter(save_dir)

    logger.info(args)
    logger.info('saved model path:{}'.format(torch.hub.get_dir()))

    if args.use_wandb:
        import wandb
        wandb.init(project='model_tuning', name=args.filename)
        wandb.config.update(args)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####################################################
    # create datasets (embeddings by pre-trained models)
    ####################################################
    datasets = create_datasets(args, device, logger)

    id_features = datasets["id_features"]
    id_loader, id_loader_test = datasets["id_loader"], datasets["id_loader_test"]
    ood_loader, ood_loader_test = datasets["ood_loader"], datasets["ood_loader_test"]
    union_loader = datasets["union_loader"]

    #########################
    # Discriminator training
    #########################
    logger.info('starting discriminative tuning')

    discriminator = Discriminator(args.num_layers_disc).to(device)
    if args.optimizer_disc == 'sgd':
        discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=args.lr_disc,
                                            momentum=0.9, weight_decay=args.weight_decay_disc)
        discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            discriminator_optimizer, T_max=args.epochs_disc)
    elif args.optimizer_disc == 'adam':
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        discriminator_scheduler = None
    else:
        raise NotImplementedError

    if args.algo != 'iswbc':
        epochs_disc = 0
    else:
        epochs_disc = args.epochs_disc

    for epo in range(epochs_disc):
        logger.info('Epoch: %d' % epo)
        discriminator.train()
        train_loss = 0
        idx = 0
        for i, data in enumerate(zip(id_loader, union_loader)):
            in_data, ood_data = data
            features, targets = in_data
            features, targets = features.to(device), targets.to(device)
            union_features, union_labels = ood_data
            union_features, union_labels = union_features.to(device), union_labels.to(device)
            true_labels_ = torch.ones(len(features), dtype=torch.float32, device=device)
            fake_labels_ = torch.zeros(len(union_features), dtype=torch.float32, device=device)
            true_logits = discriminator(features, targets).squeeze()
            fake_logits = discriminator(union_features, union_labels).squeeze()

            true_loss = F.binary_cross_entropy_with_logits(
                true_logits, true_labels_,
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits, fake_labels_
            )
            disc_loss = true_loss + fake_loss

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

            train_loss += disc_loss.item()
            idx += 1

        logger.info('[discriminator] training: Loss: {:.3f}'.format(train_loss / idx))
        writer.add_scalar('Discriminator/Loss', train_loss / idx, epo)

        if discriminator_scheduler:
            discriminator_scheduler.step()

    discriminator.eval()

    #################
    # Linear probing
    #################
    logger.info('starting Linear probing')

    EPS2 = 1e-6
    classifier = Classifier(
        hidden_layer_size=id_features.shape[-1],
        num_class=10,
        num_layers=args.num_layers_clas
    ).to(device)

    if args.optimizer_clas == 'sgd':
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr_clas,
                              momentum=0.9, weight_decay=args.weight_decay_clas)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_clas)
    else:
        raise NotImplementedError

    for epo in range(args.epochs_clas):
        logger.info('Epoch: %d' % epo)
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        weight = 0
        data_loader = id_loader if args.algo == 'bc' else union_loader
        for i, data in enumerate(data_loader):
            features, targets = data
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = classifier(features)

            with torch.no_grad():
                weight_logits = discriminator(features, targets)
                weights = torch.log(1 / (torch.sigmoid(weight_logits) + EPS2) - 1 + EPS2)
                weights = torch.exp(-weights).squeeze()
            if args.algo != 'iswbc':
                pass
                weights = torch.ones_like(weights)
            else:
                if args.clip_max:
                    weights = torch.minimum(weights, torch.ones_like(weights) * args.clip_max)
                weights = (1. - weights <= args.clip_min) * weights

            # loss = criterion(outputs, targets)
            loss = F.cross_entropy(outputs, targets, reduction='none')

            # import pdb; pdb.set_trace()
            loss = (weights * loss).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            weight += weights.mean().item()
            idx += 1

        logger.info('[{}] training: Loss: {:.3f} | Acc: {:.3f} | Weight: {:.3f}'.format(
            args.algo, train_loss / idx, 100. * correct / total, weight / idx))

        ood_loss_test, ood_acc_test, _ = test(classifier, ood_loader_test, device)
        id_loss_train, id_acc_train, _ = test(classifier, id_loader, device)
        id_loss_test, id_acc_test, _ = test(classifier, id_loader_test, device)

        logger.info('[{}] ood-test Loss: {:.3f} | Acc: {:.3f}'.format(
            args.algo, ood_loss_test, 100 * ood_acc_test))
        logger.info('[{}] id-train Loss: {:.3f} | Acc: {:.3f}'.format(
            args.algo, id_loss_train, 100 * id_acc_train))
        logger.info('[{}] id-test Loss: {:.3f} | Acc: {:.3f}'.format(
            args.algo, id_loss_test, 100 * id_acc_test))

        writer.add_scalar('Classifier/TrainLoss', train_loss / idx, epo)
        writer.add_scalar('Classifier/BatchTrainLoss', id_loss_train, epo)
        writer.add_scalar('Classifier/Weight', weight / idx, epo)
        writer.add_scalar('Classifier/TestLoss', id_loss_test / idx, epo)
        writer.add_scalar('Classifier/TestAcc', id_acc_test * 100, epo)
        writer.add_scalar('Classifier/TrainAcc', id_acc_test * 100, epo)
        writer.add_scalar('Classifier/OodTestLoss', ood_loss_test, epo)
        writer.add_scalar('Classifier/OodTestAcc', ood_acc_test * 100, epo)

        scheduler.step()

    logger.info('finishing LP')


if __name__ == '__main__':
    main(parse_args())
