from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchFewShot.LRPtools import utils
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (- targets * log_probs).mean(0).sum()
        # print(loss)
        return loss / inputs.size(2)


class FeatureMatchingLossSim(nn.Module):
    def __init__(self):
        super(FeatureMatchingLossSim, self).__init__()


    def forward(self, cosine_sim, labels):
        '''implement global pooling first'''
        # print(feature_mean.shape, features.shape, labels_test.shape)
        # print(cosine_sim.sum())
        num_class = cosine_sim.size(1)
        targets = torch.zeros(cosine_sim.size(0), cosine_sim.size(1)).scatter_(1, labels.unsqueeze(1).data.cpu(),
                                                                               1).cuda()  # the one hot encoding of the target
        intra_sim = (cosine_sim * targets).sum(-1)
        inter_sim = (cosine_sim * (torch.tensor([1]).cuda() - targets)).sum(-1)
        inter_sim = inter_sim / (num_class - 1)
        # print(inter_sim, intra_sim)
        loss_test_intra = -torch.log(intra_sim)

        loss_test_inter = -torch.log(1-inter_sim)
        # print(loss_test_intra, loss_test_inter)
        loss = loss_test_intra.mean() + loss_test_inter.mean()
        # print(loss)
        return loss

class FeatureMatchingLossDis(nn.Module):
    def __init__(self):
        super(FeatureMatchingLossDis, self).__init__()


    def forward(self, distance, labels):
        '''implement global pooling first'''
        # print(feature_mean.shape, features.shape, labels_test.shape)
        # print(cosine_sim.sum())

        num_class = distance.size(1)
        targets = torch.zeros(distance.size(0), distance.size(1)).scatter_(1, labels.unsqueeze(1).data.cpu(),
                                                                               1).cuda()  # the one hot encoding of the target
        # print(targets.size())
        intra_dis = (distance * targets).sum(-1)
        # inter_sim = (distance * (torch.tensor([1]).cuda() - targets)).sum(-1)
        # inter_sim = inter_sim / (num_class - 1)

        loss = intra_dis.mean()
        # print(loss)
        return loss

