import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import resnet
class ResnetModel(nn.Module):
    def __init__(self, args):
        super(ResnetModel, self).__init__()
        # model
        self.base = resnet.resnet50(pretrained=True)

        self.addLinear= nn.Linear(args.low_dim, args.class_num)


        self.relu = nn.ReLU()
        self.softmax = nn.functional.softmax
        self.l2= nn.functional.normalize

        #Adam optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.base.parameters()},
            {'params': self.addLinear.parameters()}],
            lr=1e-5,
            #  momentum=0.9,
            weight_decay=0.0005)
        # self.optimizer= torch.optim.SGD([
        #     {'params': self.base.parameters()},
        #     {'params': self.addLinear.parameters()}], lr=1e-2, weight_decay=5e-4, momentum=0.9)
        # the_params= [{'params':self.base.parameters(), 'lr':1e-5, 'weight_decay': 1e-5},
        #                {'params':self.addLinear.parameters(), 'lr':1e-5, 'weight_decay': 1e-5}
        #               ]
        # self.optimizer= torch.optim.Adam(the_params)
        # self.optimizer_base= torch.optim.Adam(base_params)
        # top_source = [{'params': self.sourceTop.parameters(), 'lr': 1e-5}]
        # self.optimizer_source = torch.optim.Adam(top_source)
        # top_target = [{'params': self.targetTop.parameters(), 'lr': 1e-5}]
        # self.optimizer_target = torch.optim.Adam(top_target)

        # #SGD optimizer
        # base_params = [{'params': self.base.parameters(), 'lr': 3e-3, 'momentum':0.9, 'weight_decay':1e-5},
        #                 {'params': self.addLinear.parameters(), 'lr': 3e-3, 'momentum':0.9, 'weight_decay':1e-5}]
        # self.optimizer= torch.optim.SGD(base_params)
        # top_params = [{'params': self.addTop.parameters(), 'lr': 3e-3, 'momentum': 0.9, 'weight_decay': 1e-5}]
        # self.optimizer_top= torch.optim.SGD(top_params)



    def forward(self, img, warmup= True):
        feats= self.base(img, warmup)
        logits= self.addLinear(feats)
        return logits, feats
