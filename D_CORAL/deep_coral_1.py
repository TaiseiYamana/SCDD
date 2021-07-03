from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR

from torchvision.models import alexnet, resnet50, resnet34, resnet18

sys.path.append('..')
from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from kd_losses import *

import common.vision.datasets as datasets
from common.utils.data import ForeverDataIterator
from DALoader import *
from dalib.adaptation.dcoral import DeepCoralLoss
#from coral import coral

dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='./results/Pre', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets/DA', help='path name of image dataset')

# use dataset
parser.add_argument('--dataset', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
# Domain choosen
parser.add_argument('--source', default = 'A', help='source domain(s)')
parser.add_argument('--target', default = 'W', help='target domain(s)')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=30, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--cuda', type=int, default=1)

## added parameters
parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('-i', '--iters-per-epoch', default=100, type=int,
                        help='Number of iterations per epoch')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='coral_of31_r34', help='note for this run')


# hyperparameter
parser.add_argument('--lambda_da', type=float, default=0.9, help='trade-off parameter for da loss')

args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)#./results/DA/kd_of31_A_r50_to_alex
args.img_root = os.path.join(args.img_root, args.dataset)#./datasets/DA/Office31
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		#cudnn.enabled = True
		cudnn.benchmark = False
		cudnn.deterministic = True
	logging.info("args = %s", args)
	logging.info("unparsed_args = %s", unparsed)

	# define dataset & dataloader
	_, source_train_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.source, batch_size = args.batch_size, num_workers = args.workers, train = True)
	_, source_val_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.source, batch_size = args.batch_size, num_workers = args.workers, train = False)
	_, target_train_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.target, batch_size = args.batch_size, num_workers = args.workers, train = True)
	_, target_val_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.target, batch_size = args.batch_size, num_workers = args.workers, train = False)
	source_train_iter = ForeverDataIterator(source_train_loader)
	target_train_iter = ForeverDataIterator(target_train_loader)
	n_classes = len(source_train_loader.dataset.classes)

	logging.info('----------- Network Initialization --------------')
	net = resnet34(pretrained = True)
	net.fc = nn.Linear(512, n_classes, bias = True)
	torch.nn.init.normal_(net.fc.weight, mean=0, std=5e-3)
	net.fc.bias.data.fill_(0.01)
	net = net.cuda()
	logging.info('%s', net)
	logging.info("param size = %fMB", count_parameters_in_MB(net))
	logging.info('-----------------------------------------------')

	logging.info('Saving initial parameters......')
	save_path = os.path.join(args.save_root, 'initial_r50.pth.tar')
	torch.save({
		'epoch': 0,
		'net': net.state_dict(),
		'prec@1': 0.0,
		'prec@5': 0.0,
	}, save_path)

	# define loss functions
	if args.cuda:
		coral = DeepCoralLoss().cuda()
		criterion = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterion = torch.nn.CrossEntropyLoss()

	# initialize optimizer

	optimizer = torch.optim.SGD(net.parameters(),lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay, nesterov = True)

	# define scheduler
	lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

	# warp nets and criterions for train and test
	iters = {'target':target_train_iter, 'source':source_train_iter}

	best_top1 = 0
	best_top5 = 0
	for epoch in range(1, args.epochs+1):
		#adjust_lr(optimizer, epoch)

		# train one epoch
		epoch_start_time = time.time()
		train(iters, net, optimizer, criterion, coral, lr_scheduler, epoch)

		# evaluate on testing set
		logging.info('Testing the models......')
		s_test_top1, s_test_top5 = test(source_val_loader, net, criterion, phase = 'Source')
		t_test_top1, t_test_top5 = test(target_val_loader, net, criterion, phase = 'Target')

		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		# save model
		is_best = False
		if t_test_top1 > best_top1:
			best_top1 = t_test_top1
			best_top5 = t_test_top5
			is_best = True
		logging.info('Saving models......')
		save_checkpoint({
			'epoch': epoch,
			'net': net.state_dict(),
			'prec@1': t_test_top1,
			'prec@5': t_test_top5,
		}, is_best, args.save_root)

def train(iter, net, optimizer, criterion, coral, lr_scheduler, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	da_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	source_iter = iter['source']
	target_iter = iter['target']

	net.train()

	end = time.time()
	for i in range(args.iters_per_epoch):
		source_img, source_label = next(source_iter)
		target_img, target_label = next(target_iter)

		data_time.update(time.time() - end)

		if args.cuda:
			source_img = source_img.cuda()
			source_label = source_label.cuda()
			target_img = target_img.cuda()
			target_label = target_label.cuda()

		source_out = net(source_img)
		target_out = net(target_img)

		cls_loss = criterion(source_out, source_label)
		da_loss = coral(source_out, target_out) * args.lambda_da
		loss = cls_loss + da_loss

		prec1, prec5 = accuracy(source_out, source_label, topk=(1,5))
		cls_losses.update(cls_loss.item(), source_img.size(0))
		da_losses.update(da_loss.item(), target_img.size(0))
		top1.update(prec1.item(), source_img.size(0))
		top5.update(prec5.item(), source_img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		lr_scheduler.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
					   'DA:{da_losses.val:.4f}({da_losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, args.iters_per_epoch, batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, da_losses=da_losses, top1=top1, top5=top5))
			logging.info(log_str)

def test(test_loader, net, criterion, phase):
	losses = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	net.eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			out = net(img)
			loss = criterion(out, target)

		prec1, prec5 = accuracy(out, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [losses.avg, top1.avg, top5.avg]
	logging.info('-{}- Cls Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(phase,*f_l))

	return top1.avg, top5.avg

def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()
