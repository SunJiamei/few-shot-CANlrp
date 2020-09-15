from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

sys.path.append('./torchFewShot')

# from args_tiered import argument_parser
# from args_xent import argument_parser
from args_xent_FM import argument_parser
from torchFewShot.models.net import Model, ModelwithFM, ModelwithFMLRP
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss, FeatureMatchingLoss, FeatureMatchingLRPLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate

parser = argument_parser()
args = parser.parse_args()
# original loss = loss1 + 0.5 * loss2
# FM1 loss = loss1 + 0.5 * loss2 + 0.5 * (loss3
args.save_dir = './result/miniImageNet/CAM/5-shot-seed112-resnet12-FM-flat-traintestlrp-ft'  # loss = loss1 + 0.5 * loss2 + 0.5 * (loss3 + loss4)
args.resume = './result/miniImageNet/CAM/5-shot-seed112-resnet12/best_model.pth.tar'
args.max_epoch = 40
args.LUT_lr = [(10, 0.00003), (20, 0.00001), (30, 0.000003), (40, 0.000001)]
def main(train_mode='FM_simple'):
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()
    if train_mode == 'FM_simple':
        model = ModelwithFM(scale_cls=args.scale_cls, num_classes=args.num_classes)
    elif train_mode == 'FM_LRP':
        model = ModelwithFMLRP(scale_cls=args.scale_cls, num_classes=args.num_classes)
    else:
        raise ValueError(f"the current {train_mode} is not available")
    if os.path.isfile(args.resume):
        print(f"Loading model parameters from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    criterion = CrossEntropyLoss()
    fmcriterion = FeatureMatchingLoss()
    fmlrpcriterion = FeatureMatchingLRPLoss()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)

    if use_gpu:
        model = model.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print(f'saving model to {args.save_dir}')
    print("==> Start training")

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()
        if train_mode == 'FM_simple':
            '''train with simple feature matching loss'''
            train(epoch, model, criterion, fmcriterion, optimizer, trainloader, learning_rate, use_gpu)

        elif train_mode == 'FM_LRP':
            '''train with lrp weighted feature matching loss'''
            trainwithLRP(epoch, model, criterion, fmcriterion, fmlrpcriterion, optimizer, trainloader, learning_rate,
                         use_gpu)
        else:
            raise ValueError(f"the current {train_mode} is not available")
        train_time += round(time.time() - start_train_time)

        # if epoch == 0 or epoch > (args.stepsize[0] - 1) or (epoch + 1) % 10 == 0:
        acc = test(model, testloader, use_gpu)
        is_best = acc > best_acc

        if is_best:
            best_acc = acc
            best_epoch = epoch + 1

        save_checkpoint({
            'state_dict': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

        print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, criterion, fmcriterion, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()

        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()

        ytest, cls_scores, ftrain_mean, ftrain_raw, ftest_raw = model(images_train, images_test, labels_train_1hot,
                                                                      labels_test_1hot)

        loss1 = criterion(ytest, pids.view(-1))
        loss2 = criterion(cls_scores, labels_test.view(-1))
        loss = loss1 + 0.5 * loss2
        if epoch >= 3:
            if args.nExemplars > 1:
                loss3 = fmcriterion(ftrain_mean, ftrain_raw, labels_train)
                loss += 0.5 * loss3
            loss4 = fmcriterion(ftrain_mean, ftest_raw, labels_test)
            # loss4 = attendedfm_criterion(ftrain_att_mean, ftest_att, labels_test)
            loss += 0.5 * loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
        epoch + 1, learning_rate, batch_time=batch_time,
        data_time=data_time, loss=losses))


def trainwithLRP(epoch, model, criterion, fmcriterion, fmlrpcriterion, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)

        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()

        ytest, cls_scores, ftrain_mean, ftrain_raw, ftest_raw, relevance_ftest, relevance_ftrain = model(images_train,
                                                                                                         images_test,
                                                                                                         labels_train_1hot,
                                                                                                         labels_test_1hot)

        loss1 = criterion(ytest, pids.view(-1))
        loss2 = criterion(cls_scores, labels_test.view(-1))
        loss = loss1 + 0.5 * loss2

        # if epoch >= 10 and epoch <=20:
        #     if args.nExemplars > 1:
        #         loss3 = fmcriterion(ftrain_mean, ftrain_raw, labels_train)
        #         loss += 0.5 * loss3
        # loss4 = fmcriterion(ftrain_mean, ftest_raw, labels_test)
        # loss4 = attendedfm_criterion(ftrain_att_mean, ftest_att, labels_test)
        #     loss += 0.5 * loss4
        # if epoch >= 20:
            # if args.nExemplars > 1:
            #     loss3 = fmcriterion(ftrain_mean, ftrain_raw, labels_train)
            #     loss += 0.5 * loss3
        loss4 = fmlrpcriterion(ftrain_mean, ftest_raw, relevance_ftrain, relevance_ftest, labels_test)
        loss += 0.5 * loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
        epoch + 1, learning_rate, batch_time=batch_time,
        data_time=data_time, loss=losses))


def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (images_train, labels_train, images_test, labels_test, images_test_path) in enumerate(
                testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy()  # [b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main(train_mode='FM_LRP')
