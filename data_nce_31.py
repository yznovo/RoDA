import os.path
import random
from loss.utils import noisify_multiclass_symmetric, noisify_pairflip
import torchvision.transforms
import utils.utils as utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchnet.meter import AUCMeter
import cv2
import torchvision.transforms as transforms
import json

a_path= "dataset/datasets/."
def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(scale=[0.2, 1], size=crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [val.split()[0] for val in image_list]
            labels = [int(val.split()[1]) for val in image_list]
            return images, np.array(labels)
    return images


class home_dataset(Dataset):
    def __init__(self, args, sample_ratio= None, transform=None, mode=None, pred= [], probability= [], s_noise_file='', t_noise_file='', domain= '', pseudo_label= ''):
        self.noise_ratio = args.ratio  # noise ratio
        self.transform = transform
        self.mode = mode
        self.pseudo_label= pseudo_label
        # self.transition = {}  # class transition for asymmetric noise
        self.loader = utils.rgb_loader
        if args.dataset == 'office_31':
            assert args.class_num == 31
            ss = args.dset.split('2')[0]
            tt = args.dset.split('2')[1]
            if ss == 'a':
                s = 'amazon'
            elif ss == 'w':
                s = 'webcam'
            elif ss == 'd':
                s = 'dslr'
            else:
                raise NotImplementedError
            if tt == 'a':
                t = 'amazon'
            elif tt == 'w':
                t = 'webcam'
            elif tt == 'd':
                t = 'dslr'
            else:
                raise NotImplementedError

        s_tr_path = 'dataset/datasets/office31_split/' + s + '_31_train.txt'
        s_ts_path = 'dataset/datasets/office31_split/' + s + '_31_test.txt'
        t_tr_path = 'dataset/datasets/office31_split/' + t + '_31_train.txt'
        t_ts_path = 'dataset/datasets/office31_split/' + t + '_31_test.txt'
        train_path= s_tr_path
        test_path= s_ts_path
        if domain== 't':
            train_path = t_tr_path
            test_path = t_ts_path
        train_path, test_path = open(train_path).readlines(), open(test_path).readlines()
        if self.mode == 'test':
            self.test_data, self.test_label = make_dataset(test_path, None)
        elif self.mode== 'test_tr':
            self.train_data, self.test_label= make_dataset(train_path, None)
        else:
            train_data, train_label = make_dataset(train_path, None)
            # inject noise
            noise_file= s_noise_file
            if domain== 't':
                noise_file= t_noise_file
            if args.mode== 'symmetric':
                if os.path.exists(noise_file):
                    print("load noisy label from %s" %noise_file)
                    noisy_label= json.load(open(noise_file, "r"))
                    noisy_label= np.array(noisy_label)
                else:
                    noisy_label, self.actual_noise_rate = noisify_multiclass_symmetric(
                        y_train=train_label,
                        noise=args.ratio,
                        random_state=0,
                        nb_classes=args.class_num)
                    print("save noisy label to %s" %noise_file)
                    label_list= noisy_label.tolist()
                    json.dump(label_list, open(noise_file, "w"))
            else:
                if os.path.exists(noise_file):
                    print("load noisy label from %s" %noise_file)
                    noisy_label= json.load(open(noise_file, "r"))
                    noisy_label= np.array(noisy_label)
                else:
                    noisy_label, self.actual_noise_rate = noisify_pairflip(
                        y_train=train_label,
                        noise=args.ratio,
                        random_state=0,
                        nb_classes=args.class_num)
                    print("save noisy label to %s" %noise_file)
                    label_list= noisy_label.tolist()
                    json.dump(label_list, open(noise_file, "w"))
            if self.mode == 'all':
                self.train_data = train_data
                self.noisy_label = noisy_label
                self.gt_label= train_label
            else:
                if self.mode == 'labeled':
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                elif self.mode == 'unlabeled':
                    pred_idx = (1-pred).nonzero()[0]
                    self.unlabeled_probability = [probability[i] for i in pred_idx]
                elif self.mode == 'refine':
                    pred_idx = np.where(self.pseudo_label>-1)[0]
                    self.refine_label = [self.pseudo_label[i] for i in pred_idx]
                self.train_data= [train_data[i] for i in pred_idx]
                self.part_gt_label= [train_label[i] for i in pred_idx]
                self.noisy_label= [noisy_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            path = self.train_data[index]
            data = cv2.imread(a_path+path)[:, :, ::-1]
            label, gt_label, prob = self.noisy_label[index], self.part_gt_label[index], self.probability[index]
            data1 = self.transform(data)
            data2 = self.transform(data)
            data3 = self.transform(data)
            return data1, data2, data3, label, gt_label, prob, index
        elif self.mode== 'test_tr':
            path, label= self.train_data[index], self.test_label[index]
            data = cv2.imread(a_path+path)[:, :, ::-1]
            data1= self.transform(data)
            return data1, label, index
        elif self.mode== 'refine':
            path, refine_label, gt_label= self.train_data[index], self.refine_label[index], self.part_gt_label[index]
            data = cv2.imread(a_path+path)[:, :, ::-1]
            data1= self.transform(data)
            return data1, refine_label, gt_label, index
        elif self.mode == 'unlabeled':
            path = self.train_data[index]
            noisy_label, gt_label= self.noisy_label[index], self.part_gt_label[index]
            prob= self.unlabeled_probability[index]
            data = cv2.imread(a_path+path)[:, :, ::-1]
            data1 = self.transform(data)
            data2 = self.transform(data)
            data3= self.transform(data)
            return data1, data2, data3, noisy_label, gt_label, prob, index
        elif self.mode == 'all':
            path, label, gt_label = self.train_data[index], self.noisy_label[index], self.gt_label[index]
            data = cv2.imread(a_path+path)[:, :, ::-1]
            data1 = self.transform(data)
            data2 = self.transform(data)
            return data1, label, data2, gt_label, index
        elif self.mode == 'test':
            path, label = self.test_data[index], self.test_label[index]
            data = cv2.imread(a_path+path)[:, :, ::-1]
            data1 = self.transform(data)
            return data1, label, index

    def __len__(self):
        if self.mode== 'train':
            return len(self.train_data)
        elif self.mode== 'test_tr':
            return len(self.train_data)
        elif self.mode== 'all':
            return len(self.train_data)
        elif self.mode== 'labeled':
            return len(self.train_data)
        elif self.mode== 'unlabeled':
            return len(self.train_data)
        elif self.mode== 'refine':
            return len(self.train_data)
        else:
            return len(self.test_data)

class home_loader():
    def __init__(self, args, s_noise_file, t_noise_file):
        self.ratio = args.ratio
        self.noise_mode = args.mode
        self.batch_size = args.batch_size
        self.num_workers = args.worker
        self.transform_train = image_train()
        self.transform_test = image_test()
        self.s_noise_file= s_noise_file
        self.t_noise_file= t_noise_file
    def run(self, args, mode, pred=[], prob=[], domain='', pseudo_label=''):
    #whole data set
        if mode == 'warmup':
            all_dataset = home_dataset(args, transform=self.transform_train, mode="all",
                                       s_noise_file= self.s_noise_file, t_noise_file= self.t_noise_file, domain= domain)
            all_loader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return all_loader
        # training data set
        elif mode == 'train':
            labeled_dataset = home_dataset(args, sample_ratio= None, transform=self.transform_train, mode="labeled",
                                           pred=pred, probability=prob, s_noise_file= self.s_noise_file, t_noise_file= self.t_noise_file, domain= domain)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            unlabeled_dataset = home_dataset(args, sample_ratio= None, transform=self.transform_train, mode="unlabeled", pred=pred, probability=prob, s_noise_file= self.s_noise_file, t_noise_file= self.t_noise_file, domain= domain)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader
        #testing data set
        elif mode == 'test':
            test_dataset= home_dataset(args, transform= self.transform_test, mode= 'test', s_noise_file='', t_noise_file='', domain= domain)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*2,
                shuffle=False,
            num_workers=self.num_workers)

            return test_loader

        #evaluating training data set
        elif mode == 'eval_train':
            eval_dataset = home_dataset(args, transform=self.transform_test, mode='all',
                                        s_noise_file= self.s_noise_file, t_noise_file= self.t_noise_file, domain= domain)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode== 'test_tr':
            test_tr_dataset= home_dataset(args, transform=self.transform_test, mode='test_tr', domain= domain)
            test_tr_loader= DataLoader(dataset= test_tr_dataset, batch_size=self.batch_size*2, shuffle= False, num_workers=self.num_workers)
            return test_tr_loader