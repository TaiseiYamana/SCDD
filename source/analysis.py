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

sys.path.append('..')
from dalib.adaptation.mcc import MinimumClassConfusionLoss

import common.vision.datasets as datasets
import common.vision.models as models
import common.modules as modules
from common.utils.data import ForeverDataIterator
from common.vision.transforms import ResizeImage
from common.utils.analysis import collect_feature, tsne, a_distance, confusion_matrix

from utils import AverageMeter, accuracy
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB

from kd_losses import *
from pseudo_labeling import pseudo_labeling
from split_dataset import split_dataset

import matplotlib.pyplot as plt

from pycm import ConfusionMatrix

ImageCLEF_root = "/content/drive/MyDrive/datasets/ImageCLEF"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # dataset
    if args.dataset == "ImageCLEF":
        args.img_root = ImageCLEF_root
        source_train_dataset = dataset(root=args.img_root, task=args.source, transform=train_transform)
        source_test_dataset = dataset(root=args.img_root, task=args.source, transform=test_transform)
        target_train_dataset = dataset(root=args.img_root, task=args.target, transform=train_transform)
        target_test_dataset = dataset(root=args.img_root, task=args.target, transform=test_transform)
    else:
        source_train_dataset = dataset(root=args.img_root, task=args.source, download=True, transform=train_transform)
        source_test_dataset = dataset(root=args.img_root, task=args.source, download=True, transform=test_transform)
        target_train_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=train_transform)
        target_test_dataset = dataset(root=args.img_root, task=args.target, download=True, transform=test_transform)

    # spliting train target domain datasets
    target_dataset_num = len(target_train_dataset)
    split_idx = split_dataset(target_train_dataset, 0.8, args.seed)
    target_train_dataset = dataset(root=args.img_root, task=args.target, indexs = split_idx, transform=train_transform)
    logging.info("Target train data number: Train:{}/Test:{}".format(len(split_idx),target_dataset_num))

    # dataloader
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    target_train_test_loader = DataLoader(target_train_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)
    target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    num_classes = len(source_train_loader.dataset.classes)

	# create model
    logging.info('----------- Network Initialization --------------')
    logging.info('=> using model {}'.format(args.arch))
    if ('resnet' in args.arch):
            backbone = models.__dict__[args.arch](pretrained=True)
            net = modules.Classifier(backbone, num_classes).to(device)
    else:
            net = models.__dict__[args.arch](num_classes = num_classes, pretrained = True).to(device)
    net_param = torch.load(args.model_param)
    logging.info('=> load pretrain parameter ')
    load_pretrained_model(net, net_param['net'])
    net.eval()
    for param in net.parameters():
		    param.requires_grad = False
    #logging.info('%s', tnet)
    #logging.info("param size = %fMB", count_parameters_in_MB(tnet))
    logging.info('-----------------------------------------------')

    #nets = {'snet':snet, 'tnet':tnet}

    # optimizer and lr scheduler
    #if ('resnet' in args.s_arch):
		    #params = snet.get_parameters()
    #else:
		    #params = [
            #{"params": snet.features.parameters(), "lr": 0.1 * 1},
            #{"params": snet.classifier[:6].parameters(), "lr": 0.1 * 1},
            #{"params": snet.classifier[6].parameters(), "lr": 1.0 * 1}]

    #optimizer = SGD(params, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    #lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    #source_train_iter = ForeverDataIterator(source_train_loader)
    #target_train_iter = ForeverDataIterator(target_train_loader)

    #if (args.not_select_label):
            #iters = {'source':source_train_iter,'target':target_train_iter}
    #else:
		    # select paseudo labels
    selected_idx = pseudo_labeling(args.threshold, target_train_test_loader, net)
    target_selected_dataset = dataset(root=args.img_root, task=args.target, indexs = selected_idx, transform=train_transform)
    target_train_selected_loader = DataLoader(target_selected_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers, drop_last=True)
	#target_train_selected_iter = ForeverDataIterator(target_train_selected_loader)
		    # define dict
		    #iters = {'source':source_train_iter,'target':target_train_iter, 'target_selected':target_train_selected_iter}

    # loss function
    mcc = MinimumClassConfusionLoss(temperature=args.mcc_temp)
    st = SoftTarget(args.st_temp)
    cls = torch.nn.CrossEntropyLoss()

    if args.cuda:
		    mcc = mcc.to(device)
		    st = st.to(device)
		    cls = cls.to(device)

    #if True:
        #if args.model_param != None:
            # load model paramater
            #checkpoint = torch.load(args.model_param)
            #load_pretrained_model(net, checkpoint['net'])
        # extract features from both domains
        #feature_extractor = nn.Sequential(tnet.backbone, tnet.bottleneck).to(device)
    source_feature = collect_feature(source_test_loader, net, device)
    target_feature = collect_feature(target_test_loader, net, device)
    # plot t-SNE
    tSNE_filename = os.path.join(args.save_root, 'TSNE.png')
    tsne.visualize(source_feature, target_feature, tSNE_filename)
    logging.info("Saving t-SNE to",tSNE_filename)
    # calculate A-distance, which is a measure for distribution discrepancy
    A_distance = a_distance.calculate(source_feature, target_feature, device)
    logging.info("A-distance =",A_distance)

    # plot Confusion Matrix
    CM_filename = os.path.join(args.save_root, 'ConfusionMatrix.png')
    cm_list = test(target_test_loader, net, cls, mcc, 'Target Domain')
    cm_pd = ConfusionMatrix(actual_vector=cm_list.t, predict_vector=cm_list.y)
    cm_pd.classes = target_test_dataset.CLASSES
    plt.figure()
    confusion_matrix.plot_cm(cm_pd, annot=True)
    plt.savefig(CM_filename, bbox_inches='tight')

def test(data_loader, net, cls, mcc, phase):
	cls_loss = AverageMeter()
	mcc_loss = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()
	cm_list = confusion_matrix.Cal_ConfusionMatrix(num_classes=len(data_loader.dataset.classes))

	net.eval()

	end = time.time()
	for i, (img, target, _) in enumerate(data_loader, start=1):
		img = img.cuda()
		target = target.cuda()

		with torch.inference_mode():
			out, _ = net(img)
			cls_loss = cls(out, target)
			mcc_loss = mcc(out)

		cm_list.update(out ,target)
		prec1, prec5 = accuracy(out, target, topk=(1,5))
		cls_loss.update(cls_loss.item(), img.size(0))
		mcc_loss.update(mcc_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_loss.avg, mcc_loss.avg, top1.avg, top5.avg]
	logging.info('-{}- CLS Loss: {:.4f}, MCC Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(phase,*f_l))

	return cm_list

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
    parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--model-param', default=None, type=str, help='path name of teacher model')
    parser.add_argument('--check_point', action='store_true', help='use check point parameter')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, help='weight decay (default: 1e-3)')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default= 100, type=int, help='print frequency (default: 50)')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    parser.add_argument('--cg_norm', default=5.0, type=float, help='max norm of nn.utils.clip_grad_norm_')
    # loss parameters
    parser.add_argument('--mcc_temp', default=2.5, type=float, help='parameter mcc temperature scaling')
    parser.add_argument('--st_temp', default=2.0, type=float, help='parameter soft target temperature scaling')
    parser.add_argument('--lam', default=1., type=float,
                        help='the trade-off hyper-parameter for mcc loss')
    parser.add_argument('--mu', default=1., type=float,
                        help='the trade-off hyper-parameter for soft target loss')
    # others
    parser.add_argument('--not_select_label', action='store_true')
    parser.add_argument('--not_clip_grad', action='store_true')
    parser.add_argument('--stopping_num', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.7)
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
