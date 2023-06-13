import argparse
import os
import math
import time
import random
import warnings
import numpy as np
from PIL import ImageFilter

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

from resnet import *
from utils import *

import wandb

parser = argparse.ArgumentParser(description='New SSL method Training')
# General variables
parser.add_argument('--data', metavar='DIR', default='/data/datasets/imagenette2/')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--ckpt_file', type=str, default='checkpoint.pth')
parser.add_argument('--save_dir', type=str, default='./experiments/testing/')
parser.add_argument('--workers', type=int, metavar='N', default=8)
parser.add_argument('--print_freq', type=int, default=20)
# Training variables
parser.add_argument('--epochs', type=int, metavar='N', default=100)#100
parser.add_argument('--warmup_epochs', type=int, default=0) #10
parser.add_argument('--batch_size', type=int, metavar='N', default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, metavar='W', default=0)
parser.add_argument('--schedule', type=str, default='constant') # constant, cosine
parser.add_argument('--optimizer', type=str, default='sgd') #sgd, adamw, lars
parser.add_argument('--tranform_mode', type=str, default='noaug') #'aug', 'noaug'
parser.add_argument('--zero_init_residual', action='store_true', default=False)
# SSL variables
parser.add_argument('--hidden_dim', type=int, default=2048)
parser.add_argument('--dim', type=int, default=2048)
parser.add_argument('--alpha', type=float, default=0.5)
# SEED
parser.add_argument('--seed', type=int, default=0)
# GPU
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--track_wandb', action='store_true')
parser.add_argument('--num_classes', type=int, default=10)


def main():
    args = parser.parse_args()
    
    if args.track_wandb:
        run_name = f"arch@{args.arch}_optim@{args.optimizer}_batch@{args.batch_size}_lr@{args.lr}_wd@{args.weight_decay}" \
                   f"_epoch@{args.epochs}_dim@{args.dim}_hiddendim@{args.hidden_dim}" \
                   f"_seed@{args.seed}_aug@{args.tranform_mode}_warmup@{args.warmup_epochs}_zeroinit@{args.zero_init_residual}" \
                   f"_schedule@{args.schedule}_loss@S&Malpha{args.alpha}"
        args.save_dir = os.path.join('experiments', run_name)
        experiment = wandb.init(project='new_SSL_imagenette', resume='allow', anonymous='must',
                                name=run_name, group='initial_runs') #embedding_dim_expt
        experiment.config.update(
            dict(
                arch = args.arch,
                optim = args.optimizer,
                batch_size = args.batch_size,
                lr = args.lr,
                weight_decay = args.weight_decay,
                epochs = args.epochs,
                dim = args.dim,
                hidden_dim = args.hidden_dim,
                seed = args.seed,
                aug = args.tranform_mode,
                warmup = args.warmup_epochs,
                schedule = args.schedule,
                loss = 'S&M',
                alpha = args.alpha,
                zero_init = args.zero_init_residual,
                )
        )

    print(vars(args))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_dir_models = os.path.join(args.save_dir, 'models')
    if not os.path.exists(args.save_dir_models):
        os.makedirs(args.save_dir_models)

    # Set the seed
    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args)
        
        
def main_worker(args):
    start_time_all = time.time()

    print('\nUsing Single GPU training')
    print('Use GPU: {} for training'.format(args.gpu))

    print('\nLoading dataloader ...')
    train_dataset = datasets.ImageFolder(os.path.join(args.data,'train'), Transform(mode=args.tranform_mode))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True)
    
    # Create data loaders for KNN evaluation
    args.knn_train_loader, args.knn_val_loader = get_knn_dataloaders(args, Transform)

    print('\nLoading model, optimizer')
    model = SSL_model(args)
    model = model.cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    print('\nStart training...')
    start_epoch=0
    if args.track_wandb: 
        wandb.watch(model, log='all', log_freq=1)

    for epoch in range(start_epoch, args.epochs):
        start_time_epoch = time.time()

        # train the network
        train(train_loader,model,optimizer,epoch,args)

        # print time per epoch
        end_time_epoch = time.time()
        print("Epoch time: {:.2f} minutes".format((end_time_epoch - start_time_epoch) / 60),
              "Training time: {:.2f} minutes".format((end_time_epoch - start_time_all) / 60))

    end_time_all = time.time()
    print("Total training time: {:.2f} minutes".format((end_time_all - start_time_all) / 60))


def train(loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for step, (inputs, labels) in enumerate(loader, start=epoch * len(loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        # set learning rate
        lr = adjust_learning_rate(optimizer, loader, step, args)

        # ============ forward pass ... ============
        inputs = inputs.cuda(args.gpu, non_blocking=True)
        loss = model(inputs)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss.backward()
        model.log_stuff(loss,step,lr)
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.6f}".format(
                    epoch,
                    step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return losses.avg


class SSL_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = eval(args.arch)(zero_init_residual=args.zero_init_residual)
        features_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        # projection head
        self.projection_head = nn.Sequential(nn.Linear(features_dim, args.hidden_dim, bias=False),
                                        nn.BatchNorm1d(args.hidden_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(args.hidden_dim, args.hidden_dim, bias=False),
                                        nn.BatchNorm1d(args.hidden_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(args.hidden_dim, args.dim)#, bias=False)
                                        )

    def forward(self, x):
        # encoder
        r = self.encoder(x)
        z = self.projection_head(r)
        z_norm = nn.functional.normalize(z, dim=1, p=2)

        S = torch.mm(z_norm, z_norm.T)

        if torch.sum(torch.diag(S)).item() != x.shape[0]:
            print(f"Diagonal not  {x.shape[0]}")

        M = (-1)*(0.5*S -1) * (S+1)**2

        # loss using both S and M
        alpha = self.args.alpha
        k = x.shape[0]
        combinedMS = alpha*M + (1-alpha)*(S+1)**2
        loss = (1/(k**2 - k))*torch.sum(off_diagonal(combinedMS))

        self.S = S

        return loss
    
    def log_stuff(self,loss,step, lr):
        
        # accumulate for grad flow plot
        ave_grad, layers= get_grad_flow(self.named_parameters())
        fig_grad = plt.figure(num='grad_plot', figsize=(10, 8))
        plt.plot(ave_grad, alpha=0.3, color="b")
        plt.tight_layout()
        plt.hlines(0, 0, len(ave_grad)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grad), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grad))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.yscale('log')
        plt.grid(True)


        # Log
        if self.args.track_wandb and step % self.args.print_freq == 0:
            
            # plot similarity matrix
            fig_S = plt.figure(figsize=(15, 15))
            Sim_matrix = self.S.detach().cpu().numpy()
            plt.imshow(Sim_matrix, cmap='viridis', vmin=-1, vmax=1)
            plt.colorbar()

            # plot histogram of off-diagonal values
            S_offdiag_values = list(off_diagonal(torch.tensor(Sim_matrix)).numpy())

            # plot conv1 filters
            kernels = self.encoder.conv1.weight.detach().cpu().clone()
            maxs, _ = torch.max(kernels.view(64, -1), dim=-1)
            mins, _ = torch.min(kernels.view(64, -1), dim=-1)
            kernels = (kernels - mins.view(64, 1, 1, 1)) / (maxs - mins).view(64, 1, 1, 1)
            kernel_visualization = torchvision.utils.make_grid(kernels, nrow=16, pad_value=1)

            # Get KNN performance on subset of training
            knn_acc = knn_validate(self.encoder, self.args)

            # wandb track
            wandb.log({
                "loss": loss.item(),
                "grad_flow": wandb.Image(fig_grad),
                "similarity matrix (S)": wandb.Image(fig_S),
                'S matrix offdiag histogram': wandb.Histogram(S_offdiag_values, num_bins=100),
                'conv1': wandb.Image(kernel_visualization),
                "steps": step,
                "lr": lr,
                "knn_acc": knn_acc,
                })
            
            plt.close(fig_S)
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Transform:
    def __init__(self, mode='aug'):
        if mode == 'aug':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                            saturation=0.8, hue=0.2)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                PILRandomGaussianBlur(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.228, 0.224, 0.225])
            ])
        elif mode == 'noaug':
            self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.228, 0.224, 0.225])
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        return x1
    

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def adjust_learning_rate(optimizer, loader, step, args):
    max_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)
    base_lr = args.lr * args.batch_size / 256

    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    elif args.schedule == 'cosine':
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    elif args.schedule == 'constant':
        lr = base_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def exclude_bias_and_norm(p):
    return p.ndim == 1

if __name__ == '__main__':
    main()