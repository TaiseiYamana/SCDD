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
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

sys.path.append('../..')
from dalib.adaptation.mcc import MinimumClassConfusionLoss

import common.vision.datasets as datasets
import common.vision.models as models
import common.modules as modules
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.analysis import collect_feature, tsne, a_distance

from utils import AverageMeter, accuracy
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB

from split_dataset import split_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageCLEF_root = "/content/drive/MyDrive/datasets/ImageCLEF"

def main(args):
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
    test_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # create dataset & dataloader
    dataset = datasets.__dict__[args.dataset]

    # create dataset
    if args.dataset == "ImageCLEF":
        args.img_root = ImageCLEF_root
        source_train_dataset = dataset(root=args.img_root, task=args.source, transform=train_transform)
        target_train_dataset = dataset(root=args.img_root, task=args.target, transform=train_transform)
        target_test_dataset = dataset(root=args.img_root, task=args.target, transform=test_transform)
    else:
        source_train_dataset = dataset(root=args.img_root, task=args.source, download=True, transform=train_transform)
        target_train_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=train_transform)
        target_test_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=test_transform)

    # split train target domain datasets
    target_dataset_num = len(target_train_dataset)
    split_idx = split_dataset(target_train_dataset, 0.8, args.seed)
    target_train_dataset = dataset(root=args.img_root, task=args.target, indexs = split_idx, transform=train_transform)
    logging.info("Target train data number: Train:{}/Test:{}".format(len(split_idx),target_dataset_num))

    # data loader
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    target_val_loader = DataLoader(target_train_dataset, batch_size=64,shuffle=False, num_workers=args.workers)                                     
    target_test_loader = DataLoader(target_test_dataset, batch_size=64, shuffle=False, num_workers=args.workers)

    source_train_iter = ForeverDataIterator(source_train_loader)
    target_train_iter = ForeverDataIterator(target_train_loader) 

    num_classes = len(source_train_loader.dataset.classes)

	  # create model
    logging.info('----------- Network Initialization --------------')
    logging.info('=> using pre-trained model {}'.format(args.arch))
    if args.arch == 'mobilenet_v3_small':
		    net = mobilenet_v3_small(pretrained=True)
		    net.classifier[3] = nn.Linear(1024, num_classes)
    elif args.arch == 'mobilenet_v3_large':
		    net = mobilenet_v3_large(pretrained=True)
		    net.classifier[3] = nn.Linear(1280, num_classes)  
    else:
		    backbone = models.__dict__[args.arch](pretrained=True)
		    net = modules.Classifier(backbone, num_classes)
    net = net.to(device)
    logging.info('%s', net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info('-----------------------------------------------')

    # define optimizer and lr scheduler
    if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
		    params = [
            {"params": net.features.parameters(), "lr": 0.1 * args.lr},
            {"params": net.classifier.parameters(), "lr": 1.0 * args.lr}]
    else:
		    params = net.get_parameters()  
    optimizer = SGD(params, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define optimizer and lr scheduler
    if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
		    params = [
            {"params": net.features.parameters(), "lr": 0.1 * 1},
            {"params": net.classifier.parameters(), "lr": 1.0 * 1}]
    else:
		    params = net.get_parameters()  
    optimizer = SGD(params, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)
    cls_loss = torch.nn.CrossEntropyLoss()
    if args.cuda:
		    mcc = mcc_loss.to(device)
		    cls = cls_loss.to(device)

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
        _ , _ , dis_soft, classcoufusion = test_2(target_test_loader, net, cls, mcc, args, phase = 'Target')
        print("Distributed Soft {}".format(dis_soft))
        print("MCC Vlue {}".format(classcoufusion))                
        return
	
		
    # define dict
    iters = {'source':source_train_iter, 'target':target_train_iter}

    best_top1= 0.0    
    best_top5 = 0.0

    # check point parameter load
    if (args.check_point):
		    checkpoint = torch.load(os.path.join(args.model_param))
		    load_pretrained_model(net, checkpoint['net'])
		    check_point_epoch = checkpoint['epoch']
		    optimizer.load_state_dict(checkpoint['optimizer'])
		    lr_scheduler.load_state_dict(checkpoint['scheduler'])
		    best_top1 = check_point['prec@1']       
		    best_top5 = check_point['prec@5']

    stopping_counter = 0
    for epoch in range(1, args.epochs+1):
		    # skip utill check point
		    if (args.check_point):
		    	if (check_point_epoch >= epoch) :
		    		logging.info("Skip epoch {}".format(epoch)) 
		    		continue
		    	else:                            
		    		args.check_point = False

		    # train one epoch
		    epoch_start_time = time.time()
		    train(iters, net, optimizer, lr_scheduler, cls, mcc, epoch, args)

		    # evaluate on testing set
		    logging.info('Testing the models......')
		    t_val_top1, t_val_top5 = test(target_val_loader, net, cls, mcc, args, phase = 'Target Validation')            
		    t_test_top1, t_test_top5 = test(target_test_loader, net, cls, mcc, args, phase = 'Target Test')
		
		    epoch_duration = time.time() - epoch_start_time
		    logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		    # save model
		    is_best = False
		    if t_val_top1 > best_top1:
		    	best_top1 = t_val_top1
		    	best_top5 = t_val_top5
		    	is_best = True
		    	stopping_counter = 0
		    else:
		    	stopping_counter += 1
          
		    logging.info('Saving models......')
		    save_checkpoint({'epoch': epoch,
          		            'net': net.state_dict(),
          		            'optimizer': optimizer.state_dict(),							  			
          		            'scheduler': lr_scheduler.state_dict(),
          		            'prec@1': t_val_top1,
          		            'prec@5': t_val_top5,}, 
          		            is_best, args.save_root)
            
		    if stopping_counter == 8:
		    	logging.info('Planned　Stopping Training')
		    	break
			
    # print experiment result
    checkpoint = torch.load(os.path.join(args.save_root, 'model_best.pth.tar'))		
    logging.info('{}: {}->{} \nTopAcc:{:.2f} ({} epoch)'.format(args.dataset, args.source, args.target, checkpoint['prec@1'], checkpoint['epoch']))			

def train(iters, net, optimizer, lr_scheduler, cls, mcc, epoch, args):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	mcc_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	source_iter = iters['source']
	target_iter = iters['target']

	net.train()

	end = time.time()
	for i in range(args.iters_per_epoch):
		source_img, source_label ,_ = next(source_iter)
		target_img, _, _ = next(target_iter)

		data_time.update(time.time() - end)

		if args.cuda:
			source_img = source_img.cuda()
			source_label = source_label.cuda()
			target_img = target_img.cuda()
		if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
			source_out = net(source_img)
			target_out = net(target_img)
		else:
			source_out, _= net(source_img)
			target_out, _= net(target_img)

		cls_loss = cls(source_out, source_label)
		mcc_loss = mcc(target_out)
		loss = cls_loss + mcc_loss * args.trade_off 
    
		prec1, prec5 = accuracy(source_out, source_label, topk=(1,5))
		cls_losses.update(cls_loss.item(), source_img.size(0))
		mcc_losses.update(mcc_loss.item(), target_img.size(0))
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
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, args.iters_per_epoch, batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, mcc_losses=mcc_losses, top1=top1, top5=top5))
			logging.info(log_str)


def test(test_loader, net, cls, mcc, args, phase):
	losses = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	net.eval()

	end = time.time()
	for i, (img, target, _) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
				out = net(img)
			else:
				out, _ = net(img)
			loss = cls(out, target)                
		prec1, prec5 = accuracy(out, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [losses.avg, top1.avg, top5.avg]
	logging.info('-{}- Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(phase,*f_l))

	return top1.avg, top5.avg

def test_2(test_loader, net, cls, mcc, args, phase):
	losses = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()
	mcc_losses  = AverageMeter()
	distributed_soft = AverageMeter()

	net.eval()

	end = time.time()
	for i, (img, target, _) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
				out = net(img)
			else:
				out, _ = net(img)
			loss = cls(out, target)  
			mcc_loss = mcc(out)   
		prec1, prec5 = accuracy(out, target, topk=(1,5))
		mcc_losses.update(mcc_loss.item(), img.size(0))
		softlabel = nn.functional.softmax(out / 4, dim=-1)
		softlabel_var = torch.var(softlabel, dim=-1)
		softlabel_var = torch.mean(softlabel_var)
		distributed_soft.update(softlabel_var.item())
		#print(softlabel_var.item())   
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [losses.avg, top1.avg, top5.avg]
	logging.info('-{}- Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(phase,*f_l))

	return top1.avg, top5.avg, distributed_soft.avg, mcc_losses.avg
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
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
    parser.add_argument('--model-param', default=None, type=str, help='path name of teacher model')
    parser.add_argument('--check_point', default=False, type=bool, help='use check point parameter')                     
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, help='weight decay (default: 1e-3)')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, help='print frequency (default: 50)')
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