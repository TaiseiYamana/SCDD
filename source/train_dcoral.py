# Distillation zoo
import argparse
import os
import sys
import time
import logging
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
#from torch.utils.data import DataLoader
#import torchvision.transforms as T
#import torch.nn.functional as F

sys.path.append('..')
from dalib.adaptation.dcoral import DeepCoralLoss
#from dalib.adaptation.mcc import MinimumClassConfusionLoss, ImageClassifier
import common.vision.datasets as datasets
import common.vision.models as models
#from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
#from common.utils.metric import accuracy, ConfusionMatrix
#from common.utils.meter import AverageMeter, ProgressMeter
#from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

from utils import AverageMeter, accuracy
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB

from DALoader import *
from coral import coral

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
		    torch.cuda.manual_seed(args.seed)
		    cudnn.enabled = True
		    cudnn.benchmark = True
    logging.info("args = %s", args)

	  # create dataset & dataloader
    _, source_train_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.source, batch_size = args.batch_size, num_workers = args.workers, train = True)
    _, source_val_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.source, batch_size = args.batch_size, num_workers = args.workers, train = False)
    _, target_train_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.target, batch_size = args.batch_size, num_workers = args.workers, train = True)
    _, target_val_loader = eval(args.dataset + '_loader')(root = args.img_root, task = args.target, batch_size = args.batch_size, num_workers = args.workers, train = False)
    source_train_iter = ForeverDataIterator(source_train_loader)
    target_train_iter = ForeverDataIterator(target_train_loader)

    num_classes = len(source_train_loader.dataset.classes)

	  # create model
    logging.info('----------- Network Initialization --------------')
    logging.info('=> using pre-trained model {}'.format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    net = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)
    logging.info('%s', net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info('-----------------------------------------------')

    # define optimizer and lr scheduler
    optimizer = SGD(net.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


	  # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'initial_model.pth.tar')
    torch.save({
		  'epoch': 0,
		  'net': net.state_dict(),
		  'prec@1': 0.0,
		  'prec@5': 0.0,
	  }, save_path)

    # define loss function
    dcoral = DeepCoralLoss()
    cls = torch.nn.CrossEntropyLoss()
    if args.cuda:
        dcoral = dcoral.to(device)
        cls = cls.to(device)

    if args.phase == 'analysis': 
        if args.model_param != None:
            # load model paramater
            checkpoint = torch.load(args.model_param)
            load_pretrained_model(net, checkpoint['net'])
        # extract features from both domains
        feature_extractor = nn.Sequential(net.backbone, net.bottleneck).to(device)
        source_feature = collect_feature(source_train_loader, feature_extractor, device)
        target_feature = collect_feature(target_train_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = os.path.join(args.save_root, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test': 
        if args.model_param != None:
            # load model paramater
            checkpoint = torch.load(args.model_param)
            load_pretrained_model(net, checkpoint['net'])
            print("top1acc:{:.2f}".format(checkpoint['prec@1']))
        _ , _ = test(target_val_loader, net, cls, args, phase = 'Target')
        return
	
		
    # define dict
    iters = {'target':target_train_iter, 'source':source_train_iter}

    best_top1= 0.0    
    best_top5 = 0.0
    for epoch in range(1, args.epochs+1):
		    # train one epoch
		    epoch_start_time = time.time()
		    train(iters, net, optimizer, lr_scheduler, cls, dcoral, epoch, args)

		    # evaluate on testing set
		    logging.info('Testing the models......')
		    s_test_top1, s_test_top5 = test(source_val_loader, net, cls, args, phase = 'Source')
		    t_test_top1, t_test_top5 = test(target_val_loader, net, cls, args, phase = 'Target')
		
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

def train(iters, net, optimizer, lr_scheduler, cls, dcoral, epoch, args):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	dcoral_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	source_iter = iters['source']
	target_iter = iters['target']

	net.train()

	end = time.time()
	for i in range(args.iters_per_epoch):
		source_img, source_label = next(source_iter)
		target_img, _ = next(target_iter)

		data_time.update(time.time() - end)

		if args.cuda:
			source_img = source_img.cuda()
			source_label = source_label.cuda()
			target_img = target_img.cuda()

		source_out, _= net(source_img)
		target_out, _= net(target_img)

		cls_loss = cls(source_out, source_label)
		dcoral_loss = dcoral(target_out)
		loss = cls_loss + dcoral_loss * args.trade_off 
    
		prec1, prec5 = accuracy(source_out, source_label, topk=(1,5))
		cls_losses.update(cls_loss.item(), source_img.size(0))
		dcoral_losses.update(dcoral_loss.item(), target_img.size(0))
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
					   'DCORAL:{mcc_losses.val:.4f}({dcoral_losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, args.iters_per_epoch, batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, dcoral_losses=dcoral_losses, top1=top1, top5=top5))
			logging.info(log_str)


def test(test_loader, net, cls, args, phase):
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
			out, _ = net(img)
			loss = cls(out, target)

		prec1, prec5 = accuracy(out, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [losses.avg, top1.avg, top5.avg]
	logging.info('-{}- Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(phase,*f_l))

	return top1.avg, top5.avg

if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

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
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--model-param', default=None, type=str, help='path name of teacher model')                       
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, help='weight decay (default: 1e-3)')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency (default: 50)')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    # mcc parameters
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')                         
    # others
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    args.save_root = os.path.join(args.save_root, args.note)#./results/pt_of31_A_r50
    args.img_root = os.path.join(args.img_root, args.dataset)#./datasets/Office31
    create_exp_dir(args.save_root) #save-rootの作成

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)
