import torch
import torch.nn.functional as F
import torch.nn as nn

class feature_loss(nn.Module):
    def __init__(self, s_feature, t_feature, label= 0.9, unlabel= 0.9):
        super().__init__()
        self.s_feature= s_feature.cuda()
        self.t_feature= t_feature.cuda()
        self.label= label
        self.unlabel= unlabel

    def set_param(self, epoch, args):
        self.label= 1.*epoch/args.max_epoch*(args.label_end- args.label_start)+args.label_start
        self.unlabel= 1.*epoch/args.max_epoch*(args.unlabel_end- args.unlabel_start)+args.unlabel_start

    def prepare(self, feature, the_class, type, labeled):
        length= feature.size(0)
        if type== 's':
            for i in range(length):
                self.s_feature[the_class[i],:]= self.s_feature[the_class[i],:]* 0.5+ feature[i]*(1- 0.5)
        else:
            for i in range(length):
                self.t_feature[the_class[i],:]= self.t_feature[the_class[i],:]* 0.5+ feature[i]*(1- 0.5)
        self.deta()

    def update(self, feature, the_class, type, labeled):
        length= feature.size(0)
        if labeled== True:
            if type== 's':
                for i in range(length):
                    self.s_feature[the_class[i],:]= self.s_feature[the_class[i],:]* self.label+ feature[i]*(1- self.label)
            else:
                for i in range(length):
                    self.t_feature[the_class[i],:]= self.t_feature[the_class[i],:]* self.label+ feature[i]*(1- self.label)
        else:
            if type== 's':
                for i in range(length):
                    self.s_feature[the_class[i],:]= self.s_feature[the_class[i],:]* self.unlabel+ feature[i]*(1- self.unlabel)
            else:
                for i in range(length):
                    self.t_feature[the_class[i],:]= self.t_feature[the_class[i],:]* self.unlabel+ feature[i]*(1- self.unlabel)



    def loss(self):
        return torch.mean(torch.sum(torch.abs(self.s_feature- self.t_feature), dim= 1))


    def deta(self):
        self.s_feature= self.s_feature.detach()
        self.t_feature= self.t_feature.detach()



class logits_loss(nn.Module):
    def __init__(self, s_logits, t_logits, label= 0.95, unlabel= 0.99):
        super().__init__()
        self.s_logits= s_logits.cuda()
        self.t_logits= t_logits.cuda()
        self.label= label
        self.unlabel= unlabel

    def set_param(self, epoch, args):
        self.label= 1.*epoch/args.max_epoch*(args.label_end- args.label_start)+args.label_start
        self.unlabel= 1.*epoch/args.max_epoch*(args.unlabel_end- args.unlabel_start)+args.unlabel_start

    def prepare(self, logits, the_class, type, labeled):
        length = logits.size(0)
        if type == 's':
            for i in range(length):
                self.s_logits[the_class[i]] = self.s_logits[the_class[i]] * 0.5 + logits[i] * (1 - 0.5)
        else:
            for i in range(length):
                self.t_logits[the_class[i]] = self.t_logits[the_class[i]] * 0.5 + logits[i] * (1 - 0.5)
        self.deta()

    def update(self, logits, the_class, type, labeled):
        length = logits.size(0)
        if type == 's':
            for i in range(length):
                self.s_logits[the_class[i]] = self.s_logits[the_class[i]] * self.label + logits[i] * (1 - self.label)
        else:
            for i in range(length):
                self.t_logits[the_class[i]] = self.t_logits[the_class[i]] * self.label + logits[i] * (1 - self.label)

    def loss(self):
        return torch.mean(torch.sum(torch.abs(self.s_logits- self.t_logits), dim= 1))

    def deta(self):
        self.s_logits= self.s_logits.detach()
        self.t_logits= self.t_logits.detach()





