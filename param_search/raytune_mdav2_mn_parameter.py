# Distillation zoo
import argparse
import os
import sys
import numpy as np
import random


# pytorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small


# ray tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import JupyterNotebookReporter
from ray.tune import CLIReporter
from ray.tune.suggest.bayesopt import BayesOptSearch

# local
from dalib.adaptation.mcc import MinimumClassConfusionLoss

import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
import common.modules as modules
from common.utils.data import ForeverDataIterator

from utils import AverageMeter, accuracy

from kd_losses import *

# args
architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name]))

parser = argparse.ArgumentParser(description='Train student model')

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
parser.add_argument('--t_arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')   
parser.add_argument('--t-model-param', default=None, type=str, help='path name of teacher model')
# training parameters
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, help='weight decay (default: 1e-3)')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('-i', '--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency (default: 50)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

# loss parameters
parser.add_argument('--mcc_temp', default=2.5, type=float, help='parameter mcc temperature scaling')
#parser.add_argument('--st_temp', default=2.0, type=float, help='parameter soft target temperature scaling')
parser.add_argument('--lam', default=1., type=float,
                        help='the trade-off hyper-parameter for mcc loss')
parser.add_argument('--mu', default=0.9, type=float,
                        help='the trade-off hyper-parameter for soft target loss')
# others
parser.add_argument('--cuda', type=int, default=1)
# ray tune config
parser.add_argument("--num_samples", type=int, default=10)

args = parser.parse_args()

args.img_root = os.path.join(args.img_root, args.dataset)#./datasets/Office31
ImageCLEF_root = "/content/drive/MyDrive/datasets/ImageCLEF"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

def train(models_dict, iters, loss_functions, optimizer, lr_scheduler, config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tmodel = models_dict['tmodel']
  smodel = models_dict['smodel']       
  source_iter = iters['source']
  target_iter = iters['target']
  cls = loss_functions['cls']
  da = loss_functions['da']
  kd = loss_functions['kd']

  tmodel.eval()
  smodel.train()

  for i in range(args.iters_per_epoch):
    source_img, source_label = next(source_iter)
    target_img, _ = next(target_iter)

    if torch.cuda.is_available():
      source_img = source_img.to(device)
      source_label = source_label.to(device)
      target_img = target_img.to(device)

    with torch.no_grad():
      t_target_out, _ = tmodel(target_img)

    s_source_out = smodel(source_img)
    s_target_out = smodel(target_img)

    cls_loss = cls(s_source_out, source_label)
    da_loss = da(s_target_out)
    kd_loss = kd(s_target_out, t_target_out)
    loss = cls_loss + da_loss * args.lam + kd_loss * args.mu

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

def test(model, test_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.eval()

  acc = AverageMeter()
  with torch.no_grad():
    for i, (img, label) in enumerate(test_loader, start=1):
      if torch.cuda.is_available():
	      img = img.to(device)
	      label = label.to(device)

      out = model(img)
      prec, _ = accuracy(out, label, topk=(1,5))
      acc.update(prec.item(), img.size(0))
		
  return acc.avg

def train_mdav2(config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
  # create dataset & dataloader
  dataset = datasets.__dict__[args.dataset]
    
  # dataset,dataloader
  if args.dataset == "ImageCLEF":
    source_train_dataset = dataset(root=ImageCLEF_root, task=args.source, transform=train_transform)
    target_train_dataset = dataset(root=ImageCLEF_root, task=args.target, transform=train_transform)
    target_val_dataset = dataset(root=ImageCLEF_root, task=args.target, transform=val_transform)
  else:
    source_train_dataset = dataset(root=args.img_root, task=args.source, download=True, transform=train_transform)
    target_train_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=train_transform)
    target_val_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=val_transform)
    
  source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
  target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
  target_val_loader = DataLoader(target_val_dataset, batch_size=64, shuffle=False, num_workers=args.workers)

  source_train_iter = ForeverDataIterator(source_train_loader)
  target_train_iter = ForeverDataIterator(target_train_loader)

  num_classes = len(source_train_loader.dataset.classes)

  # define model
  # Teacher model
  tbackbone = models.__dict__[args.t_arch](pretrained=True)
  tmodel = modules.Classifier(tbackbone, num_classes).to(device)
  # Student model
  
  smodel = mobilenet_v3_small(pretrained=True)
  smodel.classifier[3] = nn.Linear(1024, 12)
  smodel = smodel.to(device)

  # define optimizer and lr scheduler
  params = [
                {"params": smodel.features.parameters(), "lr": 0.1 * args.lr},
                {"params": smodel.classifier.parameters(), "lr": 1.0 * args.lr}
  ]
  optimizer = SGD(params, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
  lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

  # define loss function
  cls = torch.nn.CrossEntropyLoss()
  da = MinimumClassConfusionLoss(temperature=args.mcc_temp)
  kd = SoftTarget(config['st_temp'])   
  if torch.cuda.is_available():
    cls = cls.to(device)
    da = da.to(device)
    kd = kd.to(device)
        
  # define dict
  models_dict = {'tmodel':tmodel,'smodel':smodel}
  iters = {'target':target_train_iter, 'source':source_train_iter}
  loss_functions = {'cls':cls, 'da':da, 'kd':kd}

  for epoch in range(1,args.epochs+1):
    train(models_dict, iters, loss_functions, optimizer, lr_scheduler, config)
    target_test_acc = test(smodel, target_val_loader)

  # Send the current training result back to Tune
    print('Prec@1: {:.2f}'.format(target_test_acc))
    tune.report(accuracy=target_test_acc)

def main():
  if args.dataset != "ImageCLEF":
    dataset = datasets.__dict__[args.dataset]
    dataset(root=args.img_root, task=args.source, download=True, transform=None)
	
  config = {  "st_temp": tune.quniform(2.0, 20.0, 1)}

  scheduler = ASHAScheduler(metric="accuracy",mode="max", grace_period=5, reduction_factor=2)
  search_alg = BayesOptSearch(metric="accuracy", mode="max")
  reporter =  CLIReporter(metric_columns =["accuracy","training_iteration"])

  analysis = tune.run(train_mdav2, 
                    config=config, 
                    scheduler = scheduler, 
                    search_alg = search_alg,
                    num_samples=args.num_samples, 
                    progress_reporter = reporter,
                    resources_per_trial={'cpu': 2, 'gpu': 1})

  dfs = analysis.trial_dataframes
  ax = None  # This plots everything on the same plot
  for d in dfs.values():
    ax = d.accuracy.plot(ax=ax, legend=False)
	
  best_trial = result.get_best_trial("accuracy", "max", "last")
  print("Best trial config: {}".format(best_trial.config))
  print("Best trial final validation acc: {}".format(best_trial.last_result["accuracy"]))

if __name__ == "__main__":
  main()