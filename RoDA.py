import data_nce_home, data_nce_31, data_nce_adaptiope
import argparse
from model import RankingModel
from train_test.RoDA_train import *
import random
import os
import torch
import numpy as np
import json
import logging
from loss.asy_loss import *
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import time
from typing import Tuple
from utils import domain_loss


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResNetFc(nn.Module):
  def __init__(self, resnet_name='', use_bottleneck=True, bottleneck_dim=1, new_cls=True, class_num=1):
    super(ResNetFc, self).__init__()
    model_resnet = models.resnet50(pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.bottleneck_bn.weight.data.normal_(0, 0.005)
            self.bottleneck_bn.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
        # x = self.bottleneck_bn(x)
    y = self.fc(x)
    return  y, x


def get_logger(filename, verbosity=1, name= None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# 设置随机种子
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置超参数
def get_args():
    parser = argparse.ArgumentParser(
        description='RoDA on office-home dataset')

    parser.add_argument('--dataset', type=str, default='office_home')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=50,
                        help="maximum epoch")
    parser.add_argument('--batch_size',
                        type=int,
                        default= 16,
                        help="batch_size")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--class_num', type=int, default= 65)
    parser.add_argument('--worker', type=int, default= 6)
    parser.add_argument('--low_dim', type=int, default= 512)
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--noise_path', default='noise_file/home', type=str, help='the path of noisy label')
    parser.add_argument('--mode', type=str, default= 'symmetric')
    parser.add_argument('--warmup', type=int, default= 10)
    parser.add_argument('--dset', type=str, default= 'r2p')
    parser.add_argument('--ratio', type=float, default= 0.6)
    parser.add_argument('--threshold', type=float, default= 0.7)
    parser.add_argument('--temperature', type=float, default= 0.3)
    parser.add_argument('--lam', type=float, default= 1)
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default= '3',
                        help="device id to run")
    parser.add_argument('--beta', type=float, default= 4)
    args = parser.parse_args()
    return args


if __name__== "__main__":

    args = get_args()
    # logging
    if args.dataset== 'office_home':
        logger_path= 'train_log/home_'+args.dset+'_'+args.mode+'_'+str(args.ratio) +str(args.beta)+ '_RoDA.log'
    elif args.dataset== 'office_31':
        logger_path = 'train_log/31_' + args.dset + '_' + args.mode + '_' + str(args.ratio) + '_RoDA.log'
    elif args.dataset == 'adaptiope':
        logger_path = 'train_log/adaptiope_' + args.dset + '_' + args.mode + '_' + str(args.ratio) + '_RoDA.log'
    logger= get_logger(logger_path)
    # gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # random seed
    SEED= args.seed
    seed_torch(SEED)
    # dataloaders
    # Art Clipart Product Real_World
    if args.dataset== 'office_home':
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            args.noise_domain = 'Art'
        elif ss == 'c':
            args.noise_domain = 'Clipart'
        elif ss == 'p':
            args.noise_domain = 'Product'
        elif ss == 'r':
            args.noise_domain = 'Real_World'
        else:
            raise NotImplementedError
        s_file_path= ('%s/%s/%.1f_%s.json'%(args.noise_path, args.noise_domain, args.ratio, args.mode))
        if tt == 'a':
            args.noise_domain = 'Art'
        elif tt == 'c':
            args.noise_domain = 'Clipart'
        elif tt == 'p':
            args.noise_domain = 'Product'
        elif tt == 'r':
            args.noise_domain = 'Real_World'
        else:
            raise NotImplementedError
        t_file_path= ('%s/%s/%.1f_%s.json'%(args.noise_path, args.noise_domain, args.ratio, args.mode))
        loaders= data_nce_home.home_loader(args= args, s_noise_file= s_file_path, t_noise_file= t_file_path)
    if args.dataset== 'office_31':
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            args.noise_domain = 'amazon'
        elif ss == 'w':
            args.noise_domain = 'webcam'
        elif ss == 'd':
            args.noise_domain = 'dslr'
        else:
            raise NotImplementedError
        s_file_path = ('%s/%s/%.1f_%s.json' % (args.noise_path, args.noise_domain, args.ratio, args.mode))
        if tt == 'a':
            args.noise_domain = 'amazon'
        elif tt == 'w':
            args.noise_domain = 'webcam'
        elif tt == 'd':
            args.noise_domain = 'dslr'
        else:
            raise NotImplementedError
        t_file_path = ('%s/%s/%.1f_%s.json' % (args.noise_path, args.noise_domain, args.ratio, args.mode))
        loaders = data_nce_31.home_loader(args=args, s_noise_file=s_file_path, t_noise_file=t_file_path)
    if args.dataset== 'adaptiope':
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'p':
            args.noise_domain = 'product_images'
        elif ss == 'r':
            args.noise_domain = 'real_life'
        elif ss == 's':
            args.noise_domain = 'synthetic'
        else:
            raise NotImplementedError
        s_file_path = ('%s/%s/%.1f_%s.json' % (args.noise_path, args.noise_domain, args.ratio, args.mode))
        if tt == 'p':
            args.noise_domain = 'product_images'
        elif tt == 'r':
            args.noise_domain = 'real_life'
        elif tt == 's':
            args.noise_domain = 'synthetic'
        else:
            raise NotImplementedError
        t_file_path = ('%s/%s/%.1f_%s.json' % (args.noise_path, args.noise_domain, args.ratio, args.mode))
        loaders = data_nce_adaptiope.home_loader(args=args, s_noise_file=s_file_path, t_noise_file=t_file_path)
    # models
    net1= RankingModel.ResnetModel(args= args).cuda()
    net2= RankingModel.ResnetModel(args= args).cuda()
    # training
    s_best_acc= 0.0
    t_best_acc= 0.0
    best_acc= 0.0
    best_map= 0.0
    best_epoch= -1
    after_warmup= 0.0
    after_warmup_epoch= -1
    #s_test_tr_loader = loaders.run(args, mode='test_tr', domain='s')
    s_test_loader = loaders.run(args, mode='test', domain='s')
    s_warmup_loader = loaders.run(args, mode='warmup', domain='s')
    #t_test_tr_loader = loaders.run(args, mode='test_tr', domain='t')
    t_test_loader = loaders.run(args, mode='test', domain='t')
    t_warmup_loader = loaders.run(args, mode='warmup', domain='t')
    ## types
    s_types= np.zeros((args.class_num, args.low_dim), dtype= np.float64)
    t_types= np.zeros((args.class_num, args.low_dim), dtype= np.float64)
    ## feats
    s_feats= torch.zeros((args.class_num, args.low_dim), dtype= torch.float64)
    t_feats= torch.zeros((args.class_num, args.low_dim), dtype= torch.float64)
    dfeats_loss= domain_loss.feature_loss(s_feature=s_feats, t_feature=t_feats).cuda()
    logger.info('start training!')
    for epoch in range(args.max_epoch):
        print('Epoch:%d-------------------'%(epoch+1))
        logger.info('Epoch:{}\n'.format(epoch+1))
        # model training
        if epoch< args.warmup:
            print('Warmup Net1')
            warmup(epoch, net1, net1.optimizer, s_warmup_loader, t_warmup_loader, args)
            print('Warmup Net2')
            warmup(epoch, net2, net2.optimizer, s_warmup_loader, t_warmup_loader, args)
        else:
            # eval loader
            s_eval_loader = loaders.run(args, mode='eval_train', domain='s')
            t_eval_loader = loaders.run(args, mode='eval_train', domain='t')
            # ncnv by model 2
            s_prob2, s_prob_median2= division(net2, s_eval_loader, batch_size= args.batch_size, num_class= args.class_num, feat_dim= args.low_dim, num_neighbor= args.neighbors)
            s_pred2= (s_prob2< args.threshold)
            t_prob2, t_prob_median2= division(net2, t_eval_loader, batch_size= args.batch_size, num_class= args.class_num, feat_dim= args.low_dim, num_neighbor= args.neighbors)
            t_pred2= (t_prob2< args.threshold)
            # test ncnv acc
            s_labeled_trainloader_2, s_unlabeled_trainloader_2 = loaders.run(args, 'train', s_pred2, 1- s_prob2, domain='s')
            s_labeled_acc_2, s_labeled_total_2= division_acc(args, s_labeled_trainloader_2)
            logger.info('model2 division result:\nthe acc num of query domain={}\t the total num of query domain={}\t the acc of query domain labeled data={:.4f}\n'.format(
                    s_labeled_acc_2, s_labeled_total_2, s_labeled_acc_2/s_labeled_total_2))
            t_labeled_trainloader_2, t_unlabeled_trainloader_2 = loaders.run(args, 'train', t_pred2, 1- t_prob2, domain='t')
            t_labeled_acc_2, t_labeled_total_2= division_acc(args, t_labeled_trainloader_2)
            logger.info('model2 division result:\nthe acc num of gallery domain={}\t the total num of gallery domain={}\t the acc of gallery domain labeled data={:.4f}\n'.format(
                    t_labeled_acc_2, t_labeled_total_2, t_labeled_acc_2/t_labeled_total_2))
            # ncnv by model 1
            s_prob1, s_prob_median1= division(net1, s_eval_loader, batch_size= args.batch_size, num_class= args.class_num, feat_dim= args.low_dim, num_neighbor= args.neighbors)
            s_pred1 = (s_prob1< args.threshold)
            t_prob1, t_prob_median1= division(net1, t_eval_loader, batch_size= args.batch_size, num_class= args.class_num, feat_dim= args.low_dim, num_neighbor= args.neighbors)
            t_pred1 = (t_prob1< args.threshold)
            s_labeled_trainloader_1, s_unlabeled_trainloader_1 = loaders.run(args, 'train', s_pred1, 1 - s_prob1, domain='s')
            s_labeled_acc_1, s_labeled_total_1= division_acc(args, s_labeled_trainloader_1)
            logger.info('model1 division result:\nthe acc num of query domain={}\t the total num of query domain={}\t the acc of query domain labeled data={:.4f}\n'.format(
                    s_labeled_acc_1, s_labeled_total_1, s_labeled_acc_1 / s_labeled_total_1))
            t_labeled_trainloader_1, t_unlabeled_trainloader_1 = loaders.run(args, 'train', t_pred1, 1 - t_prob1, domain='t')
            t_labeled_acc_1, t_labeled_total_1= division_acc(args, t_labeled_trainloader_1)
            logger.info('model1 division result:\nthe acc num of gallery domain={}\t the total num of gallery domain={}\t the acc of gallery domain labeled data={:.4f}\n'.format(
                    t_labeled_acc_1, t_labeled_total_1, t_labeled_acc_1 / t_labeled_total_1))
            ##  Train---------------------------------------
            print('Train Net1')
            s_types, acc1, total1, t_types, t_acc1, t_total1, q_weights, q_acc, g_weights, g_acc= train_all(epoch, net1, net2, net1.optimizer, s_labeled_trainloader_2, s_unlabeled_trainloader_2, s_test_loader, s_types,
                                           t_labeled_trainloader_2, t_unlabeled_trainloader_2, t_types, args, dfeats_loss)
            logger.info('model1 refine result:\nthe acc num of query domain={}\t the total num of query domain={}\t the acc of query domain relabeled data={:.4f}\n'.format(
                    acc1, total1, acc1/total1))
            logger.info('model1 refine result:\nthe acc num of gallery domain={}\t the total num of gallery domain={}\t the acc of gallery domain relabeled data={:.4f}\n'.format(
                    t_acc1, t_total1, t_acc1/t_total1))
        ## test map
        cur_map= test_map(args, net1, net2, s_test_loader, t_test_loader, epoch)
        if cur_map>best_map:
            best_map= cur_map
            best_epoch= epoch+1
        # print('the best map is: %.4f in epoch: %d and the last map is :%.4f' % (best_map, best_epoch, cur_map))
        logger.info(
            ' the last map={:.4f}\t and the best_map={:.4f}\t best_epoch={}\n ' .format(cur_map, best_map, best_epoch))

