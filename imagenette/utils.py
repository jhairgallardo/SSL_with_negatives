import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torchvision.datasets as datasets

from sklearn.model_selection import train_test_split


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
    return ave_grads, layers

def get_knn_dataloaders(args, Transform):
    aux_dataset = datasets.ImageFolder(os.path.join(args.data,'train'), Transform(mode='noaug'))
    train_indices = np.arange(0, len(aux_dataset))
    train_targets = np.array(aux_dataset.targets)
    aux_train_idxs, knn_train_idxs, aux_train_targets,_ = train_test_split(train_indices,train_targets, test_size=0.1, random_state=args.seed, stratify=train_targets)
    _, knn_val_idxs = train_test_split(aux_train_idxs, test_size=0.012, random_state=args.seed, stratify=aux_train_targets)
    knn_traindataset = torch.utils.data.Subset(aux_dataset, knn_train_idxs)
    knn_valdataset = torch.utils.data.Subset(aux_dataset, knn_val_idxs)
    knn_train_loader = torch.utils.data.DataLoader(knn_traindataset,batch_size=64,shuffle=False,num_workers=2, pin_memory=False)
    knn_val_loader = torch.utils.data.DataLoader(knn_valdataset,batch_size=64,shuffle=False,num_workers=2, pin_memory=False)
    del aux_dataset, knn_traindataset, knn_valdataset
    return knn_train_loader, knn_val_loader

def knn_validate(model, args):
    model.eval()

    with torch.no_grad():
        train_features=[]
        train_targets = []
        for data in args.knn_train_loader:
            img, target = data
            img = img.cuda(args.gpu)
            target = target.cuda(args.gpu)
            feature = model(img)
            feature = torch.nn.functional.normalize(feature.squeeze(), dim=1, p=2)
            train_features.append(feature)
            train_targets.append(target)
        train_features = torch.cat(train_features, dim=0)
        train_targets = torch.cat(train_targets, dim=0)

        val_predicted_labels= []
        val_targets = []
        for data in args.knn_val_loader:
            img, target = data
            img = img.cuda(args.gpu)
            target = target.cuda(args.gpu)
            feature = model(img)
            feature = torch.nn.functional.normalize(feature.squeeze(), dim=1, p=2)
            predicted_labels = knn_predict(
                feature,
                train_features.t(),
                train_targets,
                args.num_classes,
                knn_k=20,
                knn_t=1.0,
            )
            val_predicted_labels.append(predicted_labels.cpu())
            val_targets.append(target.cpu())

    ############ Get accuracy ############
    predicted_labels = torch.cat(val_predicted_labels, dim=0)
    targets = torch.cat(val_targets, dim=0)
    top1 = (predicted_labels[:, 0] == targets).float().sum()
    acc = top1 / len(targets)
    return acc

def knn_predict(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (D, N) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.

    Returns:
        A tensor containing the kNN predictions

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels