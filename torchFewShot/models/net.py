import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchFewShot.LRPtools import lrp_modules
from torchFewShot.LRPtools import lrp_wrapper
from torchFewShot.LRPtools import utils as LRPutil
from .resnet12 import resnet12
from .cam import CAM


class Model(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores
    def extract_feature(self, x):
        x = x.unsqueeze(0)
        f = self.base(x)
        return f
    # def extract_feature(self, xtrain, xtest, ytrain, ytest):
    #     batch_size, num_train = xtrain.size(0), xtrain.size(1)
    #     num_test = xtest.size(1)
    #     K = ytrain.size(2)
    #     ytrain = ytrain.transpose(1, 2)
    #     xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
    #     xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
    #     x = torch.cat((xtrain, xtest), 0)
    #     f = self.base(x)
    #     f_h = f.size(-2)
    #     f_w = f.size(-1)
    #     ftrain = f[:batch_size * num_train]
    #     ftrain = ftrain.view(batch_size, num_train, -1)
    #     ftrain = torch.bmm(ytrain, ftrain)
    #     ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
    #     ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
    #     ftest = f[batch_size * num_train:]
    #     ftest = ftest.view(batch_size, num_test, *f.size()[1:])
    #     ftrain, ftest, _, _ = self.cam(ftrain, ftest)
    #     return ftest

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        f_h = f.size(-2)
        f_w = f.size(-1)
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest, _, _ = self.cam(ftrain, ftest)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)

        if not self.training:
            return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3)
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(batch_size * num_test, -1, f_h, f_w)
        ytest = self.clasifier(ftest)

        return ytest, cls_scores


'''LRP weighted'''

class ModelwithLRP(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(ModelwithLRP, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def test(self, ftrain, ftest):
        ftest = ftest.mean(-1)
        ftest = ftest.mean(-1)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)

        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest, _, _ = self.cam(ftrain, ftest)


        ftest_att = ftest
        ftrain = ftrain.mean(-1)
        ftrain = ftrain.mean(-1)
        ftrain_mean2 = ftrain

        if not self.training:
            return self.test(ftrain, ftest)
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)

        ftest_mean1 = ftest_att.mean(-1)
        # print(ftest.sum(), ftest_mean1.sum())
        ftest_mean2 = ftest_mean1.mean(-1)  # (b, n2, n1, c)
        # print(ftest_mean2.sum())
        ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
                                       eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
        ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
                                        eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids

        scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)

        # calculate the logits from the cosine similarity
        logits_sf = torch.softmax(scores, dim=-1)
        # initialize the relevance of each label
        relevance_logits = torch.log(LRPutil.LOGIT_BETA * (logits_sf +LRPutil.EPSILON)/ (torch.tensor([1 + LRPutil.EPSILON]).cuda() - logits_sf))

        relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(
            torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
            ftest_mean2_norm * ftrain_mean2_norm,
            relevance_logits, dim=-1)  #(b, n2, n1, c)
        relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))

        relevance_ftest_mean1 = lrp_modules.compute_lrp_mean(ftest_mean2, ftest_mean1, relevance_ftest_mul_ftrain, dim=-1)
        relevance_ftest = lrp_modules.compute_lrp_mean(ftest_mean1, ftest_att, relevance_ftest_mean1, dim=-1)  #(b, n2, n1, c, h, w)

        relevance_ftest = LRPutil.normalize_relevance(relevance_ftest,dim=3)

        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm * relevance_ftest * relevance_weight.unsqueeze(-1).unsqueeze(-1), dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])


        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3)
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(batch_size * num_test, -1, 6, 6)
        ytest = self.clasifier(ftest)
        return ytest, cls_scores



# class ModelwithLRPprune(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithLRPprune, self).__init__()
#         self.scale_cls = scale_cls
#
#         self.base = resnet12()
#         self.cam = CAM()
#
#         self.nFeat = self.base.nFeat
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(-1)
#         ftest = ftest.mean(-1)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
#
#         scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         num_test = xtest.size(1)
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0)
#         f = self.base(x)
#         f_h = f.size(-2)
#         f_w = f.size(-1)
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)
#         ftrain = torch.bmm(ytrain, ftrain)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)
#
#
#         ftest_att = ftest
#         ftrain = ftrain.mean(-1)
#         ftrain = ftrain.mean(-1)
#         ftrain_mean2 = ftrain
#
#
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)
#
#         ftest_mean1 = ftest_att.mean(-1)
#         # print(ftest.sum(), ftest_mean1.sum())
#         ftest_mean2 = ftest_mean1.mean(-1)  # (b, n2, n1, c)
#         # print(ftest_mean2.sum())
#         ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
#                                        eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
#         ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
#                                         eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids
#         # print(ftest_mean2_norm.sum(), ftrain_mean2_norm.sum())
#         scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)
#         # print('scores', scores.sum())
#         logits_sf = torch.softmax(scores, dim=-1)
#         relevance_logits = torch.log(LRPutil.LOGIT_BETA * logits_sf / (torch.tensor([1]).cuda() - logits_sf))
#
#         relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(
#             torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
#             ftest_mean2_norm * ftrain_mean2_norm,
#             relevance_logits, dim=-1)  #(b, n2, n1, c)
#         relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))
#
#         ##choose topk channels
#         top_weight, top_idx = relevance_weight.clone().detach().topk(10, -1, True, True)  # choose top 20 channels
#         mask = torch.zeros_like(relevance_weight).cuda()
#         mask = mask.scatter(-1, top_idx, 1)
#         relevance_weight = mask * relevance_weight
#         ## setting threshold
#         # mask = relevance_weight.detach().clone() <=0.8
#         # relevance_weight = mask.type_as(relevance_weight) * relevance_weight
#
#         # relevance_ftest_mul_ftrain = relevance_ftest_mul_ftrain.transpose(2,3)
#         # print(relevance_ftest_mul_ftrain.shape)
#         # relevance_ftest_mul_ftrain = torch.matmul(relevance_ftest_mul_ftrain, ytest_onehot)
#         relevance_ftest_mean1 = lrp_modules.compute_lrp_mean(ftest_mean2, ftest_mean1, relevance_ftest_mul_ftrain, dim=-1)
#         relevance_ftest = lrp_modules.compute_lrp_mean(ftest_mean1, ftest_att, relevance_ftest_mean1, dim=-1)  #(b, n2, n1, c, h, w)
#
#         relevance_ftest = LRPutil.normalize_relevance(relevance_ftest,dim=3)
#         # print(relevance_weight.min(), relevance_ftest.min())
#
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm * relevance_ftest * relevance_weight.unsqueeze(-1).unsqueeze(-1), dim=3)
#         cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
#         # print(cls_scores.shape)
#         # relevance_ftrain_mean1 = lrp_modules.compute_lrp_mean(ftrain_mean2, ftrain_mean1, relevance_ftest_mul_ftrain, dim=-1)
#         # relevance_ftrain = lrp_modules.compute_lrp_mean(ftrain_mean1, ftrain_att, relevance_ftrain_mean1, dim=-1)  # (b, n2, n1, c, h, w)
#         # relevance_ftrain = relevance_ftrain.view(batch_size, num_test, K, -1)
#         # relevance_ftrain = relevance_ftrain.transpose(2,3)
#         # relevance_ftrain = torch.matmul(relevance_ftrain, ytest)
#         # if not self.training:
#         #     return cls_scores.mean(-1).mean(-1)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)
#         ftest = ftest.transpose(2, 3)
#         ytest = ytest.unsqueeze(3)
#         ftest = torch.matmul(ftest, ytest)
#         ftest = ftest.view(batch_size * num_test, -1, f_h, f_w)
#         ytest = self.clasifier(ftest)
#         return ytest, cls_scores



# '''pnorm feature matching loss'''
#
# class ModelwithFMpnorm(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithFMpnorm, self).__init__()
#         self.scale_cls = scale_cls
#
#         self.base = resnet12()
#         self.cam = CAM()
#
#         self.nFeat = self.base.nFeat  #512
#         # self.relation_network = RelationNetwork(self.nFeat,10)
#         # self.relation_network.apply(weights_init)
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#         # self.dropout = nn.Dropout(p=
#     def max_tensor(self,X):
#         X, _ = torch.max(X, -1)
#         X, _ = torch.max(X, -1)
#         return X
#     def mean_tensor(self, X):
#         X = X.mean(-1)
#         X = X.mean(-1)
#         return X
#     def p_norm(self, centroid, features, p):
#         num_samples = features.size(1)
#         centroid = self.max_tensor(centroid)  # (bs, num_class, C)
#         centroid = F.normalize(centroid, dim=-1)
#         features = self.max_tensor(features)  # (bs, num_train, C)
#         features = F.normalize(features, dim=-1)
#         centroid = centroid.unsqueeze(1)
#         features = features.unsqueeze(2)
#         dis = centroid - features   # (bs, num_samples, num_class, C)
#         p_norm_dis = torch.norm(dis, dim=-1,p=p) * self.scale_cls
#
#         return p_norm_dis.view(self.batch_size*num_samples, -1)
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(4)
#         ftest = ftest.mean(4)  #(b, n2, n1, c)
#         # print(ftest.shape, ftrain.shape)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
#         scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1) # (b, n2, n1)
#         # logits = torch.softmax(scores,dim=-1)
#         # # print(logits.sum(dim=-1))
#         # logits = torch.log(logits/(1-logits))
#         # print(scores.shape)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         num_test = xtest.size(1)
#         self.batch_size = batch_size
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  #  batch_size * num_train, C, H, W
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0) # batch_size * num train batch_size * num_test, C, H, W
#         # print(x.size())
#         f = self.base(x)  # get the features of all the images  [train, test]
#
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)   #(batch_size, num_train, fea_num)
#         ftrain = torch.bmm(ytrain, ftrain)  #(batch_size, num_class, fea_num)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  #  average the features of one class
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
#         ftrain_mean = ftrain
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         # ftest = self.dropout(ftest)
#         ftest_raw = ftest
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)   #(b, n2, num_class, c, h, w) the feature after attention
#
#         ftrain = ftrain.mean(4)
#         ftrain = ftrain.mean(4)  # (b, n2, num_class, c)
#
#         if not self.training:
#             return self.test(ftrain, ftest)
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)  #(b, n2, num_class, c, 1, 1)
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)  # (batch_size, n2,n_class, fea_C, fea_H, fea_W)
#         cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])  # (batch_size*n2, n_class, fea_C, fea_H, fea_W)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)  # (b, n2, num_class, c*h*w)
#         ftest = ftest.transpose(2, 3) # (b, n2,  c*h*w, num_class)
#         ytest = ytest.unsqueeze(3)  #(b, n2, num_class, 1)
#         ftest = torch.matmul(ftest, ytest)  #(b, n2, c*h*w, 1)
#         ftest = ftest.view(batch_size * num_test, -1, 6, 6)
#         ytest = self.clasifier(ftest)  #(b*num_test, num_class, 1)
#
#         feature_dis = self.p_norm(ftrain_mean, ftest_raw, p=2)
#
#         return ytest, cls_scores, feature_dis
#
#
# class ModelwithFMLRPpnorm(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithFMLRPpnorm, self).__init__()
#         self.scale_cls = scale_cls
#
#         self.base = resnet12()
#         self.cam = CAM()
#
#         self.nFeat = self.base.nFeat  #512
#         # self.relation_network = RelationNetwork(self.nFeat,10)
#         # self.relation_network.apply(weights_init)
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#         # self.dropout = nn.Dropout(p=
#     def max_tensor(self,X):
#         X, _ = torch.max(X, -1)
#         X, _ = torch.max(X, -1)
#         return X
#     def p_norm(self, centroid, features, relevance_weight, p):
#         num_samples = features.size(1)
#         centroid = self.max_tensor(centroid)  # (bs, num_class, C)
#         batch_size, num_class, C, H, W = centroid.size()
#         # centroid = centroid.view(batch_size, num_class, C*H*W)
#         centroid = F.normalize(centroid, dim=-1)
#         features = self.max_tensor(features)  # (bs, num_train, C)
#         # features = features.view(batch_size, num_samples, C*H*W)
#         features = F.normalize(features, dim=-1)
#         centroid = centroid.unsqueeze(1)
#         features = features.unsqueeze(2)
#         dis = centroid - features   # (bs, num_samples, num_class, C)
#         p_norm_dis = torch.norm(dis * relevance_weight, dim=-1,p=p) * self.scale_cls
#
#         return p_norm_dis.view(batch_size*num_samples, -1)
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(4)
#         ftest = ftest.mean(4)  #(b, n2, n1, c)
#         # print(ftest.shape, ftrain.shape)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
#         scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1) # (b, n2, n1)
#         # logits = torch.softmax(scores,dim=-1)
#         # # print(logits.sum(dim=-1))
#         # logits = torch.log(logits/(1-logits))
#         # print(scores.shape)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         num_test = xtest.size(1)
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  #  batch_size * num_train, C, H, W
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0) # batch_size * num train batch_size * num_test, C, H, W
#         # print(x.size())
#         f = self.base(x)  # get the features of all the images  [train, test]
#
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)   #(batch_size, num_train, fea_num)
#         ftrain = torch.bmm(ytrain, ftrain)  #(batch_size, num_class, fea_num)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  #  average the features of one class
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
#         ftrain_mean = ftrain
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         # ftest = self.dropout(ftest)
#         ftest_raw = ftest
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)   #(b, n2, num_class, c, h, w) the feature after attention
#         ftest_att = ftest.detach()
#         ftrain = ftrain.mean(4)
#         ftrain = ftrain.mean(4)  # (b, n2, num_class, c)
#         ftrain_mean2 = ftrain.detach()
#         if not self.training:
#             return self.test(ftrain, ftest)
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)  #(b, n2, num_class, c, 1, 1)
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)  # (batch_size, n2,n_class, fea_C, fea_H, fea_W)
#         cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])  # (batch_size*n2, n_class, fea_C, fea_H, fea_W)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)  # (b, n2, num_class, c*h*w)
#         ftest = ftest.transpose(2, 3) # (b, n2,  c*h*w, num_class)
#         ytest = ytest.unsqueeze(3)  #(b, n2, num_class, 1)
#         ftest = torch.matmul(ftest, ytest)  #(b, n2, c*h*w, 1)
#         ftest = ftest.view(batch_size * num_test, -1, 6, 6)
#         ytest = self.clasifier(ftest)  #(b*num_test, num_class, 1)
#
#
#         ftest_mean1 = ftest_att.mean(4)
#         # print(ftest.sum(), ftest_mean1.sum())
#         ftest_mean2 = ftest_mean1.mean(4)  # (b, n2, n1, c)
#         # print(ftest_mean2.sum())
#         ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
#                                        eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
#         ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
#                                         eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids
#         # print(ftest_mean2_norm.sum(), ftrain_mean2_norm.sum())
#         scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)
#         # print('scores', scores.sum())
#         logits_sf = torch.softmax(scores, dim=-1)
#         relevance_logits = torch.log(LRPutil.LOGIT_BETA * logits_sf / (torch.tensor([1]).cuda() - logits_sf))
#
#         relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(
#             torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
#             ftest_mean2_norm * ftrain_mean2_norm,
#             relevance_logits, dim=-1)  #(b, n2, n1, c)
#
#         relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))
#         feature_dis = self.p_norm(ftrain_mean, ftest_raw, relevance_weight, p=2)
#
#         return ytest, cls_scores, feature_dis
#
#
# class ModelwithFMLRPattpnorm(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithFMLRPattpnorm, self).__init__()
#         self.scale_cls = scale_cls
#
#         self.base = resnet12()
#         self.cam = CAM()
#
#         self.nFeat = self.base.nFeat  #512
#         # self.relation_network = RelationNetwork(self.nFeat,10)
#         # self.relation_network.apply(weights_init)
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#         # self.dropout = nn.Dropout(p=
#     def max_tensor(self,X):
#         X, _ = torch.max(X, -1)
#         X, _ = torch.max(X, -1)
#         return X
#     def p_norm(self, centroid, features, relevance_weight, p):
#         num_samples = features.size(1)
#         centroid = self.max_tensor(centroid)  # (bs, num_class, C)
#         centroid = F.normalize(centroid, dim=-1)
#         features = self.max_tensor(features)  # (bs, num_train, C)
#         features = F.normalize(features, dim=-1)
#         dis = centroid - features   # (bs, num_samples, num_class, C)
#         p_norm_dis = torch.norm(dis * relevance_weight, dim=-1,p=p) * self.scale_cls
#
#         return p_norm_dis.view(self.batch_size*num_samples, -1)
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(4)
#         ftest = ftest.mean(4)  #(b, n2, n1, c)
#         # print(ftest.shape, ftrain.shape)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
#         scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1) # (b, n2, n1)
#         # logits = torch.softmax(scores,dim=-1)
#         # # print(logits.sum(dim=-1))
#         # logits = torch.log(logits/(1-logits))
#         # print(scores.shape)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         num_test = xtest.size(1)
#         self.batch_size = batch_size
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  #  batch_size * num_train, C, H, W
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0) # batch_size * num train batch_size * num_test, C, H, W
#         # print(x.size())
#         f = self.base(x)  # get the features of all the images  [train, test]
#
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)   #(batch_size, num_train, fea_num)
#         ftrain = torch.bmm(ytrain, ftrain)  #(batch_size, num_class, fea_num)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  #  average the features of one class
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         # ftest = self.dropout(ftest)
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)   #(b, n2, num_class, c, h, w) the feature after attention
#         ftrain_att = ftrain
#         ftest_att = ftest
#         ftrain = ftrain.mean(4)
#         ftrain = ftrain.mean(4)  # (b, n2, num_class, c)
#         ftrain_mean2 = ftrain.detach()
#         if not self.training:
#             return self.test(ftrain, ftest)
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)  #(b, n2, num_class, c, 1, 1)
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)  # (batch_size, n2,n_class, fea_C, fea_H, fea_W)
#         cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])  # (batch_size*n2, n_class, fea_C, fea_H, fea_W)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)  # (b, n2, num_class, c*h*w)
#         ftest = ftest.transpose(2, 3) # (b, n2,  c*h*w, num_class)
#         ytest = ytest.unsqueeze(3)  #(b, n2, num_class, 1)
#         ftest = torch.matmul(ftest, ytest)  #(b, n2, c*h*w, 1)
#         ftest = ftest.view(batch_size * num_test, -1, 6, 6)
#         ytest = self.clasifier(ftest)  #(b*num_test, num_class, 1)
#
#
#         ftest_mean1 = ftest_att.detach().mean(4)
#         # print(ftest.sum(), ftest_mean1.sum())
#         ftest_mean2 = ftest_mean1.mean(4)  # (b, n2, n1, c)
#         # print(ftest_mean2.sum())
#         ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
#                                        eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
#         ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
#                                         eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids
#         # print(ftest_mean2_norm.sum(), ftrain_mean2_norm.sum())
#         scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)
#         # print('scores', scores.sum())
#         logits_sf = torch.softmax(scores, dim=-1)
#         relevance_logits = torch.log(LRPutil.LOGIT_BETA * logits_sf / (torch.tensor([1]).cuda() - logits_sf))
#
#         relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(
#             torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
#             ftest_mean2_norm * ftrain_mean2_norm,
#             relevance_logits, dim=-1)  #(b, n2, n1, c)
#
#         relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))
#         feature_dis = self.p_norm(ftrain_att, ftest_att, relevance_weight, p=2)
#
#         return ytest, cls_scores, feature_dis
#
#
#
#
#
# '''cos similarity feature matching loss'''
#
#
# class ModelwithFMcos(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithFMcos, self).__init__()
#         self.scale_cls = scale_cls
#
#         self.base = resnet12()
#         self.cam = CAM()
#
#         self.nFeat = self.base.nFeat  #512
#         # self.relation_network = RelationNetwork(self.nFeat,10)
#         # self.relation_network.apply(weights_init)
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#         # self.dropout = nn.Dropout(p=
#     def max_tensor(self,X):
#         X, _ = torch.max(X, -1)
#         X, _ = torch.max(X, -1)
#         return X
#     def mean_tensor(self, X):
#         X = X.mean(-1)
#         X = X.mean(-1)
#         return X
#
#     def cosine_dis(self, centroid, features):
#         num_samples = features.size(1)
#         centroid = self.max_tensor(centroid)  # (bs, num_class, C)
#         centroid = F.normalize(centroid, dim=-1)
#         features = self.max_tensor(features)  # (bs, num_train, C)
#         features = F.normalize(features, dim=-1)
#         centroid = centroid.unsqueeze(1)
#         features = features.unsqueeze(2)
#         cosine_sim = centroid * features * 10  # (bs, num_samples, num_class, C)
#         cosine_sim = cosine_sim.sum(-1).view(self.batch_size * num_samples, -1)
#
#         return cosine_sim
#
#
#     # def cosine_dis(self, centroid, features):
#     #     num_samples = features.size(1)
#     #     centroid = self.mean_tensor(centroid)  # (bs, num_class, C)
#     #     centroid = F.normalize(centroid, dim=2)
#     #     # features = self.max_tensor(features)  # (bs, num_train, C)
#     #     features = F.normalize(features, dim=2)
#     #     centroid = centroid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#     #     features = features.unsqueeze(2)
#     #     cosine_sim = torch.sum(centroid * features, dim=3)# (bs, num_samples, num_class, C)
#     #     cosine_sim = cosine_sim.view(self.batch_size * num_samples, *cosine_sim.size()[2:]) * self.scale_cls
#     #
#     #     return cosine_sim
#
#
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(4)
#         ftest = ftest.mean(4)  #(b, n2, n1, c)
#         # print(ftest.shape, ftrain.shape)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
#         scores = self.scale_cls * torch.sum(ftest * ftrain,  dim=-1) # (b, n2, n1)
#         # logits = torch.softmax(scores,dim=-1)
#         # # print(logits.sum(dim=-1))
#         # logits = torch.log(logits/(1-logits))
#         # print(scores.shape)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         num_test = xtest.size(1)
#         self.batch_size = batch_size
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  #  batch_size * num_train, C, H, W
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0) # batch_size * num train batch_size * num_test, C, H, W
#         # print(x.size())
#         f = self.base(x)  # get the features of all the images  [train, test]
#
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)   #(batch_size, num_train, fea_num)
#         ftrain = torch.bmm(ytrain, ftrain)  #(batch_size, num_class, fea_num)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  #  average the features of one class
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
#         ftrain_mean = ftrain
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         # ftest = self.dropout(ftest)
#         ftest_raw = ftest
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)   #(b, n2, num_class, c, h, w) the feature after attention
#
#         ftrain = ftrain.mean(4)
#         ftrain = ftrain.mean(4)  # (b, n2, num_class, c)
#
#         if not self.training:
#             return self.test(ftrain, ftest)
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)  #(b, n2, num_class, c, 1, 1)
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)  # (batch_size, n2,n_class, fea_C, fea_H, fea_W)
#         cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])  # (batch_size*n2, n_class, fea_C, fea_H, fea_W)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)  # (b, n2, num_class, c*h*w)
#         ftest = ftest.transpose(2, 3) # (b, n2,  c*h*w, num_class)
#         ytest = ytest.unsqueeze(3)  #(b, n2, num_class, 1)
#         ftest = torch.matmul(ftest, ytest)  #(b, n2, c*h*w, 1)
#         ftest = ftest.view(batch_size * num_test, -1, 6, 6)
#         ytest = self.clasifier(ftest)  #(b*num_test, num_class, 1)
#
#         feature_sim = self.cosine_dis(ftrain_mean, ftest_raw)
#
#         return ytest, cls_scores, feature_sim
#
#
# class ModelwithFMLRPcos(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithFMLRPcos, self).__init__()
#         self.scale_cls = scale_cls
#         self.base = resnet12()
#         self.cam = CAM()
#         self.nFeat = self.base.nFeat  # 512
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#     def max_tensor(self,X):
#         X, _ = torch.max(X, -1)
#         X, _ = torch.max(X, -1)
#         return X
#     def cosine_dis(self, centroid, features, relevance_weight):
#         num_samples = features.size(1)
#         centroid = self.max_tensor(centroid)  # (bs, num_class, C)
#         centroid = F.normalize(centroid, dim=-1, eps=1e-12)
#         features = self.max_tensor(features)  # (bs, num_samples, C)
#         features = F.normalize(features, dim=-1, eps=1e-12)
#         centroid = centroid.unsqueeze(1)
#         features = features.unsqueeze(2)
#         cosine_sim = centroid * features * relevance_weight  # (bs, num_samples, num_class, C)
#         cosine_sim = cosine_sim.sum(-1).view(self.batch_size * num_samples, -1) * self.scale_cls
#         return cosine_sim
#
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(4)
#         ftest = ftest.mean(4)  # (b, n2, n1, c)
#         # print(ftest.shape, ftrain.shape)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim() - 1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim() - 1, eps=1e-12)
#         # print(relevance_weight.shape)
#         scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)  # (b, n2, n1)
#         # logits = torch.softmax(scores,dim=-1)
#         # print(logits.sum(dim=-1))
#         # logits = torch.log(logits/(1-logits))
#         # print(scores.shape)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         self.batch_size = batch_size
#         num_test = xtest.size(1)
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  # batch_size * num_train, C, H, W
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0)  # batch_size * num train batch_size * num_test, C, H, W
#
#         f = self.base(x)  # get the features of all the images  [train, test]
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)  # (batch_size, num_train, fea_num)
#         ftrain = torch.bmm(ytrain, ftrain)  # (batch_size, num_class, fea_num)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  # average the features of one class
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
#         ftrain_mean = ftrain
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         ftest_raw = ftest
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)  # (b, n2, num_class, c, h, w) the feature after attention
#         ftest_att = ftest
#
#         ftrain = ftrain.mean(4)
#         ftrain = ftrain.mean(4)  # (b, n2, num_class, c)
#         ftrain_mean2 = ftrain
#         if not self.training:
#             return self.test(ftrain, ftest)
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)  # (b, n2, num_class, c, 1, 1)
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm,
#                                                 dim=3)  # (batch_size, n2,n_class, fea_C, fea_H, fea_W)
#         cls_scores = cls_scores.view(batch_size * num_test,
#                                      *cls_scores.size()[2:])  # (batch_size*n2, n_class, fea_C, fea_H, fea_W)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)  # (b, n2, num_class, c*h*w)
#         ftest = ftest.transpose(2, 3)  # (b, n2,  c*h*w, num_class)
#         ytest = ytest.unsqueeze(3)  # (b, n2, num_class, 1)
#         ytest_onehot = ytest
#         ftest = torch.matmul(ftest, ytest)  # (b, n2, c*h*w, 1)
#         ftest = ftest.view(batch_size * num_test, -1, 6, 6)
#         ytest = self.clasifier(ftest)  # (b*num_test, num_class, 1)
#
#         ftest_mean1 = ftest_att.mean(4)
#         # print(ftest.sum(), ftest_mean1.sum())
#         ftest_mean2 = ftest_mean1.mean(4)  # (b, n2, n1, c)
#         # print(ftest_mean2.sum())
#         ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
#                                        eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
#         ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
#                                         eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids
#         # print(ftest_mean2_norm.sum(), ftrain_mean2_norm.sum())
#         scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)
#         # print('scores', scores.sum())
#         logits_sf = torch.softmax(scores, dim=-1)
#         relevance_logits = torch.log(LRPutil.LOGIT_BETA * logits_sf / (torch.tensor([1]).cuda() - logits_sf))
#
#         relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(
#             torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
#             ftest_mean2_norm * ftrain_mean2_norm,
#             relevance_logits, dim=-1)  #(b, n2, n1, c)
#
#         relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))
#
#         feature_cosine_sim = self.cosine_dis(ftrain_mean, ftest_raw, relevance_weight)  #(bs, num_test, num_class)
#         # relevance_ftest_mul_ftrain = relevance_ftest_mul_ftrain.transpose(2,3)
#         # print(relevance_ftest_mul_ftrain.shape)
#         # relevance_ftest_mul_ftrain = torch.matmul(relevance_ftest_mul_ftrain, ytest_onehot)
#         # relevance_ftest_mean1 = lrp_modules.compute_lrp_mean(ftest_mean2, ftest_mean1, relevance_ftest_mul_ftrain, dim=-1)
#         # relevance_ftest = lrp_modules.compute_lrp_mean(ftest_mean1, ftest_att, relevance_ftest_mean1, dim=-1)  #(b, n2, n1, c, h, w)
#         # relevance_ftest = relevance_ftest.view(batch_size, num_test, K, -1)
#         # relevance_ftest = relevance_ftest.transpose(2,3)
#         # relevance_ftest = torch.matmul(relevance_ftest, ytest)
#         #
#         # relevance_ftrain_mean1 = lrp_modules.compute_lrp_mean(ftrain_mean2, ftrain_mean1, relevance_ftest_mul_ftrain, dim=-1)
#         # relevance_ftrain = lrp_modules.compute_lrp_mean(ftrain_mean1, ftrain_att, relevance_ftrain_mean1, dim=-1)  # (b, n2, n1, c, h, w)
#         # relevance_ftrain = relevance_ftrain.view(batch_size, num_test, K, -1)
#         # relevance_ftrain = relevance_ftrain.transpose(2,3)
#         # relevance_ftrain = torch.matmul(relevance_ftrain, ytest)
#
#
#         return  ytest, cls_scores, feature_cosine_sim
#
#
# class ModelwithFMLRPAttendedcos(nn.Module):
#     def __init__(self, scale_cls, num_classes=64):
#         super(ModelwithFMLRPAttendedcos, self).__init__()
#         self.scale_cls = scale_cls
#         self.base = resnet12()
#         self.cam = CAM()
#         self.nFeat = self.base.nFeat  # 512
#         self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
#     def max_tensor(self,X):
#         X, _ = torch.max(X, -1)
#         X, _ = torch.max(X, -1)
#         return X
#     def cosine_dis(self, centroid, features, relevance_weight):
#         num_samples = features.size(1)
#         centroid = self.max_tensor(centroid)  # (bs, num_class, C)
#         centroid = F.normalize(centroid, dim=-1, eps=1e-12)
#         features = self.max_tensor(features)  # (bs, num_samples, C)
#         features = F.normalize(features, dim=-1, eps=1e-12)
#         cosine_sim = centroid * features * relevance_weight  # (bs, num_samples, num_class, C)
#         cosine_sim = cosine_sim.sum(-1).view(self.batch_size * num_samples, -1) * self.scale_cls
#         # print(cosine_sim.sum())
#         return cosine_sim
#
#     def test(self, ftrain, ftest):
#         ftest = ftest.mean(4)
#         ftest = ftest.mean(4)  # (b, n2, n1, c)
#         # print(ftest.shape, ftrain.shape)
#         ftest = F.normalize(ftest, p=2, dim=ftest.dim() - 1, eps=1e-12)
#         ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim() - 1, eps=1e-12)
#         scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)  # (b, n2, n1)
#         # logits = torch.softmax(scores,dim=-1)
#         # print(logits.sum(dim=-1))
#         # logits = torch.log(logits/(1-logits))
#         # print(scores.shape)
#         return scores
#
#     def forward(self, xtrain, xtest, ytrain, ytest):
#
#         batch_size, num_train = xtrain.size(0), xtrain.size(1)
#         self.batch_size = batch_size
#         num_test = xtest.size(1)
#         K = ytrain.size(2)
#         ytrain = ytrain.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
#
#         xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  # batch_size * num_train, C, H, W
#         xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
#         x = torch.cat((xtrain, xtest), 0)  # batch_size * num train batch_size * num_test, C, H, W
#
#         f = self.base(x)  # get the features of all the images  [train, test]
#         ftrain = f[:batch_size * num_train]
#         ftrain = ftrain.view(batch_size, num_train, -1)  # (batch_size, num_train, fea_num)
#         ftrain = torch.bmm(ytrain, ftrain)  # (batch_size, num_class, fea_num)
#         ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  # average the features of one class
#         ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
#         ftest = f[batch_size * num_train:]
#         ftest = ftest.view(batch_size, num_test, *f.size()[1:])
#         ftrain, ftest, _, _ = self.cam(ftrain, ftest)  # (b, n2, num_class, c, h, w) the feature after attention
#         ftrain_att = ftrain
#         ftest_att = ftest
#
#
#         ftrain = ftrain.mean(4)
#         ftrain = ftrain.mean(4)  # (b, n2, num_class, c)
#         ftrain_mean2 = ftrain
#
#
#         ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
#         ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
#         ftrain_norm = ftrain_norm.unsqueeze(4)
#         ftrain_norm = ftrain_norm.unsqueeze(5)  # (b, n2, num_class, c, 1, 1)
#         cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm,
#                                                 dim=3)  # (batch_size, n2,n_class, fea_C, fea_H, fea_W)
#         cls_scores = cls_scores.view(batch_size * num_test,
#                                      *cls_scores.size()[2:])  # (batch_size*n2, n_class, fea_C, fea_H, fea_W)
#
#         ftest = ftest.view(batch_size, num_test, K, -1)  # (b, n2, num_class, c*h*w)
#         ftest = ftest.transpose(2, 3)  # (b, n2,  c*h*w, num_class)
#         ytest = ytest.unsqueeze(3)  # (b, n2, num_class, 1)
#         ftest = torch.matmul(ftest, ytest)  # (b, n2, c*h*w, 1)
#         ftest = ftest.view(batch_size * num_test, -1, 6, 6)
#         ytest = self.clasifier(ftest)  # (b*num_test, num_class, 1)
#
#         ftest_mean1 = ftest_att.detach().mean(4)
#         # print(ftest.sum(), ftest_mean1.sum())
#         ftest_mean2 = ftest_mean1.mean(4)  # (b, n2, n1, c)
#         # print(ftest_mean2.sum())
#         ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim() - 1,
#                                        eps=1e-12)  # this is the attended test features, each test sample corresponds to a set of features
#         ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim() - 1,
#                                         eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids
#         # print(ftest_mean2_norm.sum(), ftrain_mean2_norm.sum())
#         scores = self.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1)  #(b, n2, n1)
#         # print('scores', scores.sum())
#         logits_sf = torch.softmax(scores, dim=-1)
#         relevance_logits = torch.log(LRPutil.LOGIT_BETA * logits_sf / (torch.tensor([1]).cuda() - logits_sf))
#
#         # print(logits.shape)
#         # print(ytest_onehot.shape)
#         # print('logits', logits_sf.sum())
#         # relevance_logits = logits * ytest_onehot.squeeze(-1)
#         # print(relevance_logits.sum())
#         relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(
#             torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
#             ftest_mean2_norm * ftrain_mean2_norm,
#             relevance_logits, dim=-1)  #(b, n2, n1, c)
#
#         relevance_weight = LRPutil.normalize_relevance(relevance_ftest_mul_ftrain.squeeze(-1))
#         # print(relevance_weight.sum())
#         if not self.training:
#             return self.test(ftrain_mean2, ftest_att)
#         feature_cosine_sim = self.cosine_dis(ftrain_att, ftest_att, relevance_weight)  #(bs, num_test, num_class)
#
#
#
#         return  ytest, cls_scores, feature_cosine_sim