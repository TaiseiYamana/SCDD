# import library
# jeneral
import argparse
import os
import sys
import time
import logging
import numpy as np
import random

# pytorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

# local
sys.path.append('../..')
from dalib.adaptation.mcc import MinimumClassConfusionLoss

import common.vision.datasets as datasets
import common.vision.models as models
import common.modules as modules
from common.utils.data import ForeverDataIterator
from common.utils.analysis import collect_feature, tsne, a_distance

from utils import AverageMeter, accuracy
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name]))

parser = argparse.ArgumentParser(description='pretrain Teacher net')
# root path
parser.add_argument('--save_root', type=str, default='./results/PT', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--note', type=str, default='pt_of31_A2W_r50', help='note for this run') #office31_source_pretrain
# dataset parameters
parser.add_argument('-d', '--dataset', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
parser.add_argument('-s', '--source', default = 'A', help='source domain(s)')
parser.add_argument('-t', '--target', default = 'W', help='target domain(s)')
# model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
parser.add_argument('--model-param', default=None, type=str, help='path name of teacher model')                       
# training parameters
#parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
#parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, help='weight decay (default: 1e-3)')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
# mcc parameters
parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling')
parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')  
# number of epoch
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('-i', '--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')

# another config
parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency (default: 50)')                       
parser.add_argument('--cuda', type=int, default=1)

# ray tune config
parser.add_argument("num_samples", type=int, default = 10)

args = parser.parse_args()

args.save_root = os.path.join(args.save_root, args.note)#./results/pt_of31_A_r50
args.img_root = os.path.join(args.img_root, args.dataset)#./datasets/Office31
create_exp_dir(args.save_root) #save-rootの作成

def train(net, iters, loss_functions, optimizer, lr_scheduler):
    net.train()
    
    source_iter = iters['source']
    target_iter = iters['target']
    
    cls = loss_functions['cls']
    mcc = loss_functions['mcc']   
    
    for i in range(args.iters_per_epoch):
        source_img, source_label = next(source_iter)
        target_img, _ = next(target_iter)
        
        if torch.cuda.is_available():
            source_img = source_img.to(device)
            source_label = source_label.to(device)
            target_img = target_img.to(device)
           
    optimizer.zero_grad()
    
    source_out, _= net(source_img)
    target_out, _= net(target_img)
    
    cls_loss = cls(source_out, source_label)
    mcc_loss = mcc(target_out)
    loss = cls_loss + mcc_loss * args.trade_off 
    
    loss.backward()    
    loptimizer.step()
    lr_scheduler.step()

def test(net, test_loader, cls, args, phase):
	net.eval()
  
	for i, (img, target) in enumerate(test_loader, start=1):
		if torch.cuda.is_available():
			img = img.to(divece)
			target = target.to(divece)

		with torch.no_grad():
			out, _ = net(img)
			loss = cls(out, target)

		prec1, _ = accuracy(out, target, topk=(1,5))

	return top1.avg

def train_mcc(config):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
		    torch.cuda.manual_seed(args.seed)
		    cudnn.enabled = True
		    cudnn.benchmark = True
    logging.info("args = %s", args)
    
    # augumentation
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    
    # dataset,dataloader
    dataset = datasets.__dict__[args.dataset]
    source_train_dataset = dataset(root=args.img_root, task=args.source, download=True, transform=train_transform)
    target_train_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=train_transform)
    target_val_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=val_transform)
    
    source_train_loader = DataLoader(source_train_dataset, batch_size=config["batch_size"],
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=config["batch_size"],
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    target_val_loader = DataLoader(target_val_dataset, batch_size=64, shuffle=False, num_workers=args.workers)

    source_train_iter = ForeverDataIterator(source_train_loader)
    target_train_iter = ForeverDataIterator(target_train_loader)

    # define model
    backbone = models.__dict__[args.arch](pretrained=True)
    net = modules.Classifier(backbone, num_classes).to(device)
    
    # define optimizer and lr scheduler
    optimizer = SGD(net.get_parameters(), config["lr"], momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: config["lr"] * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)
    cls_loss = torch.nn.CrossEntropyLoss()
    if args.cuda:
		    mcc = mcc_loss.to(device)
		    cls = cls_loss.to(device)

     # define dict
    iters = {'target':target_train_iter, 'source':source_train_iter}
    loss_functions = {'cls':cls, 'mcc':mcc}

    for epoch in range(1, args.epochs+1):
		    train(iters, net, optimizer, lr_scheduler, cls, mcc, epoch, args)
		    t_test_top1 = test(target_val_loader, net, cls, args, phase = 'Target')
		    tune.report(mean_accuracy=t_test_top1)
        
main():
    config = {"lr": tune.loguniform(1e-4, 1e-1),
              "batch_size": tune.choice([32, 64, 128])}
    scheduler = ASHAScheduler(metric="mean_accuracy",mode="max")
    reporter = CLIReporter(metric_columns =["mean_accuracy"])
    analysis = tune.run(train_mcc,
                        num_samples = args.num_samples, 
                        scheduler = scheduler, 
                        config = config, 
                        progress_reporter = reporter, 
                        resources_per_trial = {"gpu": 1})
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)
    best_trial = result.get_best_trial("mean_accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation acc: {}".format(best_trial.last_result["mean_accuracy"]))

if __name__ == "__main__":
    main()
    
    
