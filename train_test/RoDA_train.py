import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import math
from utils.utils import *
from criteria import mAP
from loss.loss import JAN
from sklearn.mixture import GaussianMixture
import random
from sklearn.metrics.pairwise import cosine_similarity

mse = torch.nn.MSELoss().cuda()
mae = torch.nn.L1Loss().cuda()


# CE= nn.CrossEntropyLoss(reduction= 'none')

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


conf_penalty = NegEntropy()
CE = nn.CrossEntropyLoss(reduction='none').cuda()
CEloss = nn.CrossEntropyLoss().cuda()


def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in range (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def euclidean_distance(a, b, b_square):
    a_square = torch.sum(torch.pow(a, 2), dim=1, keepdim=True)


    ab_inner_product = torch.matmul(a, b.t())


    distance_square = a_square - 2 * ab_inner_product + b_square.t()


    distance = torch.sqrt(distance_square)
    # print(distance.shape)
    return distance


def mixup(inputs, targets, alpha):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    idx = torch.randperm(inputs.size(0))
    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target


def cal_dist(types, feats):
    diff = feats[:, np.newaxis, :] - types
    # np.sqrt()
    distances = np.sum(diff ** 2, axis=2)
    return distances


## test acc
def division_acc(args, labeled_loader):
    num_iter = len(labeled_loader.dataset) // args.batch_size
    labeled_iter = iter(labeled_loader)
    label_acc = 0
    label_total = 0
    for idx in range(num_iter):
        try:
            data = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            data = next(labeled_iter)
        noise_label = data[3]
        gt_label = data[4]
        correct = (noise_label == gt_label)
        label_acc += torch.sum(correct)
        label_total += noise_label.shape[0]

    return label_acc, label_total


## load feature
def getFeature(net, net2, trainloader, testloader, feat_dim, num_class):
    transform_bak = trainloader.dataset.transform
    trainloader.dataset.transform = testloader.dataset.transform
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=16, shuffle=False, num_workers=8)

    trainFeatures = torch.rand(len(trainloader.dataset), feat_dim).t().cuda()
    trainLogits = torch.rand(len(trainloader.dataset), num_class).t().cuda()
    trainW = torch.rand(len(trainloader.dataset)).cuda()
    trainNoisyLabels = torch.zeros(len(trainloader.dataset)).cuda()
    trainGtLabels = torch.zeros(len(trainloader.dataset)).cuda()
    trainIndex = torch.rand(len(trainloader.dataset)).cuda()

    for batch_idx, (inputs, _, _, labels, gt_labels, w, index) in enumerate(temploader):
        batchSize = inputs.size(0)
        logits, features = net(inputs.cuda())
        logits2, features2 = net2(inputs.cuda())

        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = (features + features2).data.t()
        trainLogits[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = (logits + logits2).data.t()
        trainNoisyLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = labels.cuda().data
        trainGtLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = gt_labels.cuda().data
        trainW[batch_idx * batchSize:batch_idx * batchSize + batchSize] = w.data
        trainIndex[batch_idx * batchSize:batch_idx * batchSize + batchSize] = index.data

    # trainFeatures= normalize(trainFeatures.t()).t()
    # trainLogits= normalize(trainLogits.t()).t()

    trainFeatures = trainFeatures.detach().cpu().numpy().T
    trainLogits = trainLogits.detach().cpu().numpy().T
    trainNoisyLabels = trainNoisyLabels.detach().cpu().numpy()
    trainGtLabels = trainGtLabels.detach().cpu().numpy()
    trainW = trainW.detach().cpu().numpy()
    trainIndex = trainIndex.detach().cpu().numpy()

    trainloader.dataset.transform = transform_bak
    return trainFeatures, trainLogits, trainNoisyLabels, trainGtLabels, trainW, trainIndex


## warm up
def warmup(epoch, net, optimizer, dataloader, t_dataloader, args):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    num_iter2 = (len(t_dataloader.dataset) // t_dataloader.batch_size) + 1
    num_iter = max(num_iter, num_iter2)
    s_iter = iter(dataloader)
    t_iter = iter(t_dataloader)
    for batch_idx in range(num_iter):
        try:
            s_data = next(s_iter)
        except:
            s_iter = iter(dataloader)
            s_data = next(s_iter)
        try:
            t_data = next(t_iter)
        except:
            t_iter = iter(t_dataloader)
            t_data = next(t_iter)
        s_inputs, s_labels, s_inputs2 = s_data[0].cuda(), s_data[1].cuda(), s_data[2].cuda()
        t_inputs, t_labels, t_inputs2 = t_data[0].cuda(), t_data[1].cuda(), t_data[2].cuda()
        optimizer.zero_grad()
        s_outputs, _ = net(s_inputs, False)
        t_outputs, _ = net(t_inputs, False)
        loss = CEloss(s_outputs, s_labels) + CEloss(t_outputs, t_labels)
        if args.mode == 'asymmetric':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(s_outputs + t_outputs)
            L = loss + penalty
        elif args.mode == 'symmetric':
            L = loss
        L.backward()
        optimizer.step()


def division(net, eval_loader, num_class, batch_size, feat_dim=512, num_neighbor=5):
    net.eval()

    # loading given samples
    trainFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    trainLogits = torch.rand(len(eval_loader.dataset), num_class).t().cuda()
    trainNoisyLabels = torch.rand(len(eval_loader.dataset)).cuda()
    for batch_idx, (inputs, labels, _, _, _) in enumerate(eval_loader):
        batchSize = inputs.size(0)
        logits, features = net(inputs.cuda())
        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
        trainLogits[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = logits.data.t()
        trainNoisyLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = labels.cuda().data

    trainFeatures = normalize(trainFeatures.t())
    trainLogits = trainLogits.t()
    trainNoisyLabels = trainNoisyLabels

    num_batch = math.ceil(float(trainFeatures.size(0)) / batch_size)  # 返回大于或等于参数的最小整数
    LI_collection = []
    for batch_idx in range(num_batch):
        features = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        self_logits = logits[batch_idx * batchSize:batch_idx * batchSize + batchSize]
        # self_probs= F.softmax(self_logits, dim=-1)
        noisy_labels = trainNoisyLabels[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(features, trainFeatures.t())
        dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  # set self-contrastive samples to -1
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        neighbors = neighbors.view(-1)
        neigh_logits = trainLogits[neighbors]
        neigh_probs = F.softmax(neigh_logits, dim=-1)
        M, _ = features.shape
        given_labels = torch.full(size=(M, num_class), fill_value=0.0001).cuda()
        given_labels.scatter_(dim=1, index=torch.unsqueeze(noisy_labels.long(), dim=1), value=1 - 0.0001)
        self_labels = given_labels
        given_labels = given_labels.repeat(1, num_neighbor).view(-1, num_class)
        sver = js_div(neigh_probs, given_labels)
        # sver_self= CE(self_logits, self_labels)
        # print(sver_self)
        LI_collection += sver.view(-1, num_neighbor).mean(dim=1).cpu().numpy().tolist()
        # print(sver.view(-1, num_neighbor).mean(dim=1))

    LI_collection = np.array(LI_collection)
    return LI_collection, np.median(LI_collection)


## renew types
def renew_types(cur_types, types):
    for i in range(cur_types.shape[0]):
        if np.all(cur_types[i] == 0.0):
            cur_types[i] = types[i]
        elif not np.all(types[i] == 0.0):
            cur_types[i] = cur_types[i] * 0.5 + types[i] * 0.5
    return cur_types


## prototypes
def cal_types(net, net2, labeled_loader, test_loader, low_dim, class_num):
    net.eval()
    net2.eval()
    cur_types = np.zeros((class_num, low_dim), dtype=np.float64)
    labeled_feature, _, labels, _, _, _ = getFeature(net, net2, labeled_loader, test_loader, low_dim, class_num)
    for i in range(class_num):

        class_indices = np.where(labels == i)[0]
        if len(class_indices) != 0:

            class_mean = np.mean(labeled_feature[class_indices], axis=0)
            cur_types[i] = class_mean
    return cur_types


## training
def train_all(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_loader, testloader, s_types,
              t_labeled_trainloader, t_unlabeled_loader, t_types, args, domain_feats):
    ## labeled protypes
    s_cur_types = cal_types(net, net2, labeled_trainloader, testloader, args.low_dim, args.class_num)
    s_mix_types = torch.tensor(renew_types(s_cur_types, s_types)).cuda()
    ## renew types
    s_types = s_mix_types.cpu().numpy().copy()
    t_cur_types = cal_types(net, net2, t_labeled_trainloader, testloader, args.low_dim, args.class_num)
    t_mix_types = torch.tensor(renew_types(t_cur_types, t_types)).cuda()
    ## renew types
    t_types = t_mix_types.cpu().numpy().copy()
    ## training...........
    net.train()
    net2.eval()
    acc = 0
    total = 1
    t_acc = 0
    t_total = 1
    s_actual_num = 0
    t_actual_num = 0
    ## unlabeled loader
    unlabeled_iter = iter(unlabeled_loader)
    ## labeled loader
    labeled_train_iter = iter(labeled_trainloader)
    ## iter num
    l_num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    un_num_iter = (len(unlabeled_loader.dataset) // args.batch_size) + 1
    num_iter = max(l_num_iter, un_num_iter)
    ## t domain
    # unlabeled loader
    t_unlabeled_iter = iter(t_unlabeled_loader)
    # labeled loader
    t_labeled_train_iter = iter(t_labeled_trainloader)
    ## iter num
    tl_num_iter = (len(t_labeled_trainloader.dataset) // args.batch_size) + 1
    tun_num_iter = (len(t_unlabeled_loader.dataset) // args.batch_size) + 1
    t_num_iter = max(tl_num_iter, tun_num_iter)
    ## 迭代次数
    num_iter = max(num_iter, t_num_iter)
    q_weights = np.zeros((len(unlabeled_loader.dataset), 1))
    q_acc = np.zeros((len(unlabeled_loader.dataset), 1), dtype=int)
    g_weights = np.zeros((len(t_unlabeled_loader.dataset), 1))
    g_acc = np.zeros((len(t_unlabeled_loader.dataset), 1), dtype=int)
    for batch_idx in range(num_iter):
        # labeled data
        try:
            inputs_xw, inputs_xw2, inputs_xs, labels_x, _, w_x, _ = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_xw, inputs_xw2, inputs_xs, labels_x, _, w_x, _ = next(labeled_train_iter)
        # unlabeled data
        try:
            inputs_u, inputs_u2, _, labels_u, gtlabels_u, _, index_u = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            inputs_u, inputs_u2, _, labels_u, gtlabels_u, _, index_u = next(unlabeled_iter)
        # t domain
        # labeled data
        try:
            t_inputs_xw, t_inputs_xw2, t_inputs_xs, t_labels_x, _, t_w_x, _ = next(t_labeled_train_iter)
        except:
            t_labeled_train_iter = iter(t_labeled_trainloader)
            t_inputs_xw, t_inputs_xw2, t_inputs_xs, t_labels_x, _, t_w_x, _ = next(t_labeled_train_iter)
        # unlabeled data
        try:
            t_inputs_u, t_inputs_u2, _, t_labels_u, t_gtlabels_u, _, t_index_u = next(t_unlabeled_iter)
        except:
            t_unlabeled_iter = iter(t_unlabeled_loader)
            t_inputs_u, t_inputs_u2, _, t_labels_u, t_gtlabels_u, _, t_index_u = next(t_unlabeled_iter)

        ## labeled data
        inputs_xw = inputs_xw.cuda()
        labels_x = labels_x.cuda()
        labels_x_detach = labels_x.detach().clone()
        ## label data loss
        labels_x = F.one_hot(labels_x, args.class_num).cuda()
        logits_x, _ = net(inputs_xw)
        loss = (-torch.sum(F.log_softmax(logits_x, dim=1) * labels_x, dim=1).sum()) / len(labels_x)
        ## domain loss
        inputs_xw2 = inputs_xw2.cuda()
        _, feats_x2 = net(inputs_xw2)
        feats_x2 = l2norm(feats_x2, dim=-1)
        domain_feats.update(feats_x2, labels_x_detach, 's', True)
        ## t domain
        t_inputs_xw = t_inputs_xw.cuda()
        t_labels_x = t_labels_x.cuda()
        t_labels_x_detach = t_labels_x.detach().clone()
        t_labels_x = F.one_hot(t_labels_x, args.class_num).cuda()
        t_logits_x, _ = net(t_inputs_xw)
        t_loss = (-torch.sum(F.log_softmax(t_logits_x, dim=1) * t_labels_x, dim=1).sum()) / len(t_labels_x)
        ## domain loss
        t_inputs_xw2 = t_inputs_xw2.cuda()
        _, t_feats_x2 = net(t_inputs_xw2)
        t_feats_x2 = l2norm(t_feats_x2, dim=-1)
        domain_feats.update(t_feats_x2, t_labels_x_detach, 't', True)

        ## relabel for unlabeled data
        inputs_u = inputs_u.cuda()
        logits_u, feats_u = net(inputs_u)
        ## 根据cur_types得到labels
        s_mix_types = s_mix_types.to(feats_u.dtype)
        scores = torch.matmul(feats_u, s_mix_types.t())
        mix_soft_labels = F.softmax(scores, dim=1)
        ## 准确率
        gtlabels_u = gtlabels_u.cuda()
        total += gtlabels_u.shape[0]
        mix_hard_labels = torch.argmax(mix_soft_labels, dim=1, keepdim=True).reshape(-1, 1)
        acc += torch.sum(mix_hard_labels.reshape(1, -1) == gtlabels_u)
        q_acc1 = (mix_hard_labels.cpu().numpy().reshape(1, -1) == gtlabels_u.cpu().numpy()).astype(int)
        q_acc[index_u.numpy().flatten()] = q_acc1.reshape(-1, 1)
        ## soft label ce
        ce = -torch.sum(mix_soft_labels * F.log_softmax(logits_u / args.temperature, dim=1), dim=1)
        ## 计算weight
        s_weight = (1 + args.beta) / (args.beta + torch.exp(args.beta * ce))
        s_weight = s_weight.unsqueeze(1).detach()
        s_actual_num += len(s_weight[s_weight > 0.5])
        q_weights[index_u.numpy().flatten()] = s_weight.cpu().numpy()
        ## soft_label ce
        Lu = torch.mul((-torch.sum(mix_soft_labels * F.log_softmax(logits_u, dim=1), dim=1).reshape(-1, 1)),
                       s_weight).sum() / (s_weight.size(0))
        ## domain loss
        inputs_u2 = inputs_u2.cuda()
        _, feats_u2 = net(inputs_u2)
        feats_u2 = l2norm(feats_u2, dim=-1)
        feats_u2 = feats_u2 * s_weight.detach()
        domain_feats.update(feats_u2, mix_hard_labels.view(-1).detach(), 's', False)

        ##relabel data for t domain
        t_inputs_u = t_inputs_u.cuda()
        t_logits_u, t_feats_u = net(t_inputs_u)

        t_mix_types = t_mix_types.to(t_feats_u.dtype)
        t_scores = torch.matmul(t_feats_u, t_mix_types.t())
        t_mix_soft_labels = F.softmax(t_scores, dim=1)

        t_gtlabels_u = t_gtlabels_u.cuda()
        t_total += t_gtlabels_u.shape[0]
        t_mix_hard_labels = torch.argmax(t_mix_soft_labels, dim=1, keepdim=True).reshape(-1, 1)
        t_mix_hard_labels = t_mix_hard_labels.cuda()
        t_acc += torch.sum(t_mix_hard_labels.reshape(1, -1) == t_gtlabels_u)
        g_acc1 = (t_mix_hard_labels.cpu().numpy().reshape(1, -1) == t_gtlabels_u.cpu().numpy()).astype(int)
        g_acc[t_index_u.numpy().flatten()] = g_acc1.reshape(-1, 1)

        t_ce = -torch.sum(t_mix_soft_labels * F.log_softmax(t_logits_u / args.temperature, dim=1), dim=1)
        ## weight
        t_weight = (1 + args.beta) / (args.beta + torch.exp(args.beta * t_ce))
        t_weight = t_weight.unsqueeze(1).detach()
        t_actual_num += len(t_weight[t_weight > 0.5])
        g_weights[t_index_u.numpy().flatten()] = t_weight.cpu().numpy()
        ## soft_label ce
        t_Lu = torch.mul((-torch.sum(t_mix_soft_labels * F.log_softmax(t_logits_u, dim=1), dim=1).reshape(-1, 1)),
                         t_weight).sum() / (t_weight.size(0))
        ## domain loss
        t_inputs_u2 = t_inputs_u2.cuda()
        _, t_feats_u2 = net(t_inputs_u2)
        t_feats_u2 = l2norm(t_feats_u2, dim=-1)
        t_feats_u2 = t_feats_u2 * t_weight.detach()
        domain_feats.update(t_feats_u2, t_mix_hard_labels.view(-1).detach(), 't', False)

        # all losses
        loss += t_loss
        if batch_idx % 50 == 0:
            print('labeled:')
            print(loss)
            print('unlabeled:')
            print(Lu+t_Lu)
        if epoch > args.warmup:
            loss += (epoch / args.max_epoch) * (Lu + t_Lu)
            feats_loss = domain_feats.loss()
            d_loss = feats_loss
            loss += args.lam * d_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        domain_feats.deta()
    return s_types, acc, total, t_types, t_acc, t_total, q_weights, q_acc, g_weights, g_acc


def weights_original(loader, args, model1):
    model1.eval()

    len_c = len(loader.dataset)
    weights = np.zeros((len_c, 1), dtype=float)
    accs = np.zeros((len_c, 1), dtype=int)
    for idx, data in enumerate(loader):
        img = data[0].cuda()
        label = data[3].cuda()
        gt_label = data[4]
        index = data[6]
        logits, _ = model1(img)

        ce = CEloss(logits, label)
        ce_c = ce.cpu().detach().numpy()
        weight_c = (1 + args.beta) / (args.beta + np.exp(args.beta * ce_c))
        weights[index.numpy().flatten()] = weight_c
        acc_c = (label.cpu().numpy() == gt_label.numpy()).astype(int)
        accs[index.numpy().flatten()] = acc_c.reshape(-1, 1)
    return weights, accs


def weights_acc(q_weights, q_acc, g_weights, g_acc):
    print(q_weights[:20])
    print(g_weights[:20])
    print(q_weights.shape)
    print(q_acc.shape)
    print(g_weights.shape)
    print(g_acc.shape)
    stop


def calculate_feature(model1, model2, dataloader, args):
    model1.eval()
    model2.eval()
    total = len(dataloader.dataset)
    all_feature = np.zeros((total, args.low_dim), dtype=float)
    all_label = np.zeros((total, 1), dtype=int)
    for idx, data in enumerate(dataloader):
        img = data[0].cuda()
        gt_label = data[1]
        index = data[2]
        _, feats1 = model1(img)
        _, feats2 = model2(img)
        feats = (feats1 + feats2) / 2
        feats = feats.cpu().detach().numpy().reshape(-1, args.low_dim)
        index = index.numpy().flatten()
        gt_label = gt_label.numpy().reshape(-1, 1)
        all_feature[index] = feats
        all_label[index] = gt_label
    return all_feature, all_label


# top-5
def retrieve_top_k(query_loader, gallery_loader, model1, model2, k, args):
    model1.eval()
    model2.eval()
    gallery_feats = np.zeros((len(gallery_loader.dataset), args.low_dim), dtype=float)
    query_feats = np.zeros((len(query_loader.dataset), args.low_dim), dtype=float)
    top_k = np.zeros((len(query_loader.dataset), 1), dtype=int)

    with torch.no_grad():
        for idx, data in enumerate(gallery_loader):
            img = data[0].cuda()
            label = data[1]
            index = data[2]
            _, feats1 = model1(img)
            _, feats2 = model2(img)
            feats = (feats1 + feats2) / 2
            gallery_feats[index.numpy().flatten()] = feats.cpu().detach().numpy()
        for idx, data in enumerate(query_loader):
            img = data[0].cuda()
            index = data[2]
            _, feats1 = model1(img)
            _, feats2 = model2(img)
            feats = (feats1 + feats2) / 2
            query_feats[index.numpy().flatten()] = feats.cpu().detach().numpy()

    print(query_feats.shape)
    print(gallery_feats.shape)
    similarity_matrix = np.dot(query_feats, gallery_feats.T)  # 矩阵相乘
    norm_matrix1 = np.linalg.norm(query_feats, axis=1, keepdims=True)  # 计算 matrix1 中每个向量的范数
    norm_matrix2 = np.linalg.norm(gallery_feats, axis=1, keepdims=True)  # 计算 matrix2 中每个向量的范数


    similarity_matrix /= np.dot(norm_matrix1, norm_matrix2.T)
    top_k_index = np.argsort(similarity_matrix)[:, -k:]
    return top_k_index


def test(epoch, net1, net2, test_loader):
    net1.eval()
    net2.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs)
            outputs2, _ = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    return acc


def test_map1(args, net1, train_loader, test_loader, epoch):
    net1.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, feature3 = net1(inputs)
            # _, feature2 = net2(inputs)
            # feature3= (feature1+ feature2)/2
            if batch_idx == 0:
                np_feature = feature3.cpu().detach().numpy()
                np_label = targets.cpu().detach().numpy()
            else:
                np_feature = np.concatenate((np_feature, feature3.cpu().detach().numpy()), axis=0)
                np_label = np.concatenate((np_label, targets.cpu().detach().numpy()), axis=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, feature3 = net1(inputs)
            # _, feature2 = net2(inputs)
            # feature3 = (feature1 + feature2) / 2

            if batch_idx == 0:
                test_feature = feature3.cpu().detach().numpy()
                test_label = targets.cpu().detach().numpy()
            else:
                test_feature = np.concatenate((test_feature, feature3.cpu().detach().numpy()), axis=0)
                test_label = np.concatenate((test_label, targets.cpu().detach().numpy()), axis=0)
    map = mAP.test_target(test_feature, test_label, np_feature, np_label)
    return map


def test_map(args, net1, net2, train_loader, test_loader, epoch):
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, feature1 = net1(inputs)
            _, feature2 = net2(inputs)
            feature3 = (feature1 + feature2) / 2
            if batch_idx == 0:
                np_feature = feature3.cpu().detach().numpy()
                np_label = targets.cpu().detach().numpy()
            else:
                np_feature = np.concatenate((np_feature, feature3.cpu().detach().numpy()), axis=0)
                np_label = np.concatenate((np_label, targets.cpu().detach().numpy()), axis=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, feature1 = net1(inputs, False)
            _, feature2 = net2(inputs, False)
            feature3 = (feature1 + feature2) / 2

            if batch_idx == 0:
                test_feature = feature3.cpu().detach().numpy()
                test_label = targets.cpu().detach().numpy()
            else:
                test_feature = np.concatenate((test_feature, feature3.cpu().detach().numpy()), axis=0)
                test_label = np.concatenate((test_label, targets.cpu().detach().numpy()), axis=0)
    map = mAP.test_target(test_feature, test_label, np_feature, np_label)
    return map


def test_knn(args, model1, model2, train_loader, test_loader, epoch):
    from sklearn.neighbors import KNeighborsClassifier
    neigh1 = KNeighborsClassifier(n_neighbors=1)
    # neigh5 = KNeighborsClassifier(n_neighbors=5)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, gt_labels,
                        index) in enumerate(train_loader):
            inputs = inputs.cuda()
            _, feats = model1(inputs)
            _, feats2 = model2(inputs)
            feats = feats.cpu().numpy()
            feats2 = feats2.cpu().numpy()
            feats = (feats + feats2) / 2
            if batch_idx == 0:
                train_feats = feats
                train_targets = gt_labels
            else:
                train_feats = np.concatenate((train_feats, feats), 0)
                train_targets = np.concatenate((train_targets, gt_labels), 0)

    with torch.no_grad():
        for batch_idx, (inputs, gt_labels,
                        index) in enumerate(test_loader):
            inputs = inputs.cuda()
            _, feats = model1(inputs)
            _, feats2 = model2(inputs)
            feats = feats.cpu().numpy()
            feats2 = feats2.cpu().numpy()
            feats = (feats + feats2) / 2
            if batch_idx == 0:
                test_feats = feats
                test_targets = gt_labels
            else:
                test_feats = np.concatenate((test_feats, feats), 0)
                test_targets = np.concatenate((test_targets, gt_labels), 0)

    neigh1.fit(train_feats, train_targets)
    # neigh5.fit(train_feats, train_targets)

    acc1 = (neigh1.predict(test_feats)
            == test_targets).sum() / len(test_targets)
    # acc5 = (neigh5.predict(test_feats)
    #         == test_targets).sum() / len(test_targets)

    return acc1
