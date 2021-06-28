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

sys.path.append('../..')
from dalib.adaptation.mcc import MinimumClassConfusionLoss, ImageClassifier
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

from kd_losses import *

from DALoader import *

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
    logging.info('Initialize Teacher Model')
    logging.info('=> using pre-trained model {}'.format(args.t_arch))
    tbackbone = models.__dict__[args.t_arch](pretrained=True)
    tnet = ImageClassifier(tbackbone, num_classes, bottleneck_dim=args.t_bottleneck_dim).to(device)
    checkpoint = torch.load(args.t_model_param)
    load_pretrained_model(tnet, checkpoint['net'])
    tnet.eval()
    for param in tnet.parameters():
		    param.requires_grad = False
    logging.info('%s', tnet)
    logging.info("param size = %fMB", count_parameters_in_MB(tnet))

    logging.info('Initialize Student Model')
    logging.info('=> using pre-trained model {}'.format(args.s_arch))
    sbackbone = models.__dict__[args.s_arch](pretrained=True)
    snet = Classifier(sbackbone, num_classes).to(device)
    #snet = ImageClassifier(sbackbone, num_classes, bottleneck_dim=args.s_bottleneck_dim).to(device)
    logging.info('%s', snet)
    logging.info("param size = %fMB", count_parameters_in_MB(snet))
    logging.info('-----------------------------------------------')

    # define optimizer and lr scheduler
    optimizer = SGD(snet.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


	  # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'initial_model.pth.tar')
    torch.save({
		  'epoch': 0,
		  'net': snet.state_dict(),
		  'prec@1': 0.0,
		  'prec@5': 0.0,
	  }, save_path)

    # define loss function
    mcc = MinimumClassConfusionLoss(temperature=args.mcc_temp)
    st = SoftTarget(args.st_temp)
    cls = torch.nn.CrossEntropyLoss()

    if args.cuda:
		    mcc = mcc.to(device)
		    st = st.to(device)
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
        _ , _ = test(target_val_loader, snet, cls, args, phase = 'Target')
        return


    # define dict
    iters = {'target':target_train_iter, 'source':source_train_iter}
    nets = {'snet':snet, 'tnet':tnet}

    best_top1= 0.0
    best_top5 = 0.0
    for epoch in range(1, args.epochs+1):
		    # train one epoch
		    epoch_start_time = time.time()
		    train(iters, nets, optimizer, lr_scheduler, cls, mcc, st, epoch, args)

		    # evaluate on testing set
		    logging.info('Testing the models......')
		    s_test_top1, s_test_top5 = test(source_val_loader, snet, cls, args, phase = 'Source')
		    t_test_top1, t_test_top5 = test(target_val_loader, snet, cls, args, phase = 'Target')
		
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
          'net': snet.state_dict(),
          'prec@1': t_test_top1,
          'prec@5': t_test_top5,
          }, is_best, args.save_root)

def train(iters, nets, optimizer, lr_scheduler, cls, mcc, st, epoch, args):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	mcc_losses  = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	source_iter = iters['source']
	target_iter = iters['target']

	snet = nets['snet']
	tnet = nets['tnet']

	snet.train()

	end = time.time()
	for i in range(args.iters_per_epoch):
		source_img, source_label = next(source_iter)
		target_img, _ = next(target_iter)

		data_time.update(time.time() - end)

		if args.cuda:
			source_img = source_img.cuda()
			source_label = source_label.cuda()
			target_img = target_img.cuda()

		s_source_out, _= snet(source_img)
		s_target_out, _= snet(target_img)
		t_target_out, _= tnet(target_img)

		cls_loss = cls(s_source_out, source_label)
		mcc_loss = mcc(s_target_out)
		kd_loss = st(s_target_out, t_target_out)
		loss = cls_loss + mcc_loss * args.lam + kd_loss * args.mu

		prec1, prec5 = accuracy(s_source_out, source_label, topk=(1,5))
		cls_losses.update(cls_loss.item(), source_img.size(0))
		mcc_losses.update(mcc_loss.item(), target_img.size(0))
		kd_losses.update(kd_loss.item(), target_img.size(0))
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
					   'MCC:{mcc_losses.val:.4f}({mcc_losses.avg:.4f})  '
					   'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, args.iters_per_epoch, batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, mcc_losses=mcc_losses, kd_losses=kd_losses, top1=top1, top5=top5))
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
    parser.add_argument('--t_arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--s_arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--t-bottleneck-dim', default=1024, type=int,
                        help='Dimension of bottleneck')
    #parser.add_argument('--s-bottleneck-dim', default=256, type=int,
     #                   help='Dimension of bottleneck')
    parser.add_argument('--t-model-param', default=None, type=str, help='path name of teacher model')
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
    # loss parameters
    parser.add_argument('--mcc_temp', default=2.0, type=float, help='parameter mcc temperature scaling')
    parser.add_argument('--st_temp', default=4.0, type=float, help='parameter soft target temperature scaling')
    parser.add_argument('--lam', default=1., type=float,
                        help='the trade-off hyper-parameter for mcc loss')
    parser.add_argument('--mu', default=0.9, type=float,
                        help='the trade-off hyper-parameter for soft target loss')
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
