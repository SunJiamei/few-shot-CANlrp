from torchFewShot.LRPtools import lrp_wrapper
from torchFewShot.LRPtools import lrp_modules
from torchFewShot.LRPtools import lrp_presets
from torchFewShot.LRPtools import utils
import torch.nn as nn
import torch.nn.functional as F
from torchFewShot.models import net
from torchFewShot.data_manager import DataManager
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot
import argparse
import torch
import os
import sys
import torch.backends.cudnn as cudnn
import os.path as osp
import json
from PIL import Image
import matplotlib.pyplot as plt
import torchFewShot.LRPtools.utils as LRPutil
# from args_xent_kernalsize1 import argument_parser
from args_xent import argument_parser
import numpy as np
def project(x):
    absmax = np.max(np.abs(x))
    x = 1.0 * x / absmax
    if np.sum(x < 0):
        x = (x + 1) / 2
    else:
        x = x
    return x * 255

def get_class_label(labels, image_roots, class_to_readable):
    label_to_classlabel = {}
    batch_size = labels.size()[0]
    num_test = labels.size()[1]
    print(batch_size, num_test)
    for i in range(batch_size):
        for j in range(num_test):
            image_root = image_roots[j][i]
            class_index = image_root.split('/')[-2]
            readable_label = class_to_readable[class_index]
            label = int(labels[i][j].detach().cpu().numpy())
            label_to_classlabel[label] = readable_label
        break
    return label_to_classlabel
def get_xtrain_idx(labels, index):
     labels = labels.cpu().numpy()
     # print(labels)
     return np.where(labels==index)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    args.test_batch = 1
    args.nTestNovel = 5
    args.nExemplars = 1
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    args.resume = 'please define your model path'
    with open('path to /class_to_readablelabel.json', 'r') as f:
        class_to_readable = json.load(f)
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
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

    model = net.Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    # load the model
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint['state_dict'])
    print("Loaded checkpoint from '{}'".format(args.resume))

    if use_gpu:
        model = model.cuda()
    model.eval()
    base = model.base
    preset = lrp_presets.SequentialPresetA()
    lrp_wrapper.add_lrp(base,preset=preset)
    for batch_idx, (images_train, labels_train, images_test, labels_test, images_test_pathes) in enumerate(testloader):
        # print(len(images_test_pathes))
        # print(len(images_test_pathes[0]))
        # print(images_test_pathes)

        if batch_idx >0:
            break
        else:
            label_to_classlabel = get_class_label(labels_test,images_test_pathes, class_to_readable)
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)
            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()
            # cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            batch_size, num_train = images_train.size(0), images_train.size(1)
            num_test = images_test.size(1)
            K = labels_train_1hot.size(2)
            ytrain = labels_train_1hot.transpose(1, 2)  # batchsize, num_class(the one hot), num_train
            xtrain = images_train.view(-1, images_train.size(2), images_train.size(3), images_train.size(4))  #  batch_size * num_train, C, H, W
            xtest = images_test.view(-1, images_test.size(2), images_test.size(3), images_test.size(4))
            x = torch.cat((xtrain, xtest), 0) # batch_size * num train batch_size * num_test, C, H, W  (30, 3,84,84)  30 = 25 + 5
            f = base(x)  # get the features of all the images  [train, test]  (30, 512,6,6)

            f_height, f_width = f.size()[-2], f.size()[-1]
            ftrain = f[:batch_size * num_train]
            ftrain = ftrain.view(batch_size, num_train, -1)   #(batch_size, num_train, fea_num)
            ftrain = torch.bmm(ytrain, ftrain)  #(batch_size, num_class, fea_num)
            ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  #  average the features of one class
            ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # (batch_size, num_class, fea_C, fea_H, fea_W)
            ftest = f[batch_size * num_train:]
            ftest = ftest.view(batch_size, num_test, *f.size()[1:])

            ftrain_attended, ftest_attended, ftrain_attention, ftest_attention = model.cam(ftrain, ftest)

            ftrain_mean1 = ftrain_attended.mean(-1)
            ftrain_mean2 = ftrain_mean1.mean(-1)  # (b, n2, num_class, c)
            ftest_mean1 = ftest_attended.mean(-1)
            ftest_mean2 = ftest_mean1.mean(-1)  #(b, n2, n1, c)
            ftest_mean2_norm = F.normalize(ftest_mean2, p=2, dim=ftest_mean2.dim()-1, eps=1e-12)      # this is the attended test features, each test sample corresponds to a set of features
            ftrain_mean2_norm = F.normalize(ftrain_mean2, p=2, dim=ftrain_mean2.dim()-1, eps=1e-12)  # this is the attended centroid, each test sample corresponds to a set of centroids
            # print(ftest.shape, ftrain.shape)
            scores = model.scale_cls * torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1) # (b, n2, n1)
            logits_sf = torch.softmax(scores,dim=-1)
            print(logits_sf)
            logits = torch.log(LRPutil.LOGIT_BETA * logits_sf/(torch.tensor([1.]).cuda()-logits_sf))
            print(logits)
            preds, preds_index = torch.max(logits,dim=-1)
            '''relevance backpropagate'''

            save_folder = 'your save folder/explain_heatmaps/{}shot_episode{}'.format(args.nExemplars,batch_idx)
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            for cls in range(args.nKnovel):
                explain_preds = logits.narrow(2, cls, 1)
                print('explain_preds', explain_preds.shape)
                relevance_hard_logits = torch.zeros_like(logits)
                relevance_hard_logits[:,:,cls] = logits[:,:,cls]

                relevance_logits = relevance_hard_logits
                relevance_ftest_mul_ftrain = lrp_modules.compute_lrp_sum(torch.sum(ftest_mean2_norm * ftrain_mean2_norm, dim=-1),
                                                                         ftest_mean2_norm * ftrain_mean2_norm,
                                                                         relevance_logits, dim=-1)

                # print(relevance_ftest_mul_ftrain)
                # print(relevance_ftest_mul_ftrain.shape)
                #the we treat the ftrain as weight and backpropagate the relevance only to the ftest
                # with torch.no_grad():
                relevance_ftest_mean2 = relevance_ftest_mul_ftrain
                # print(ftest_mean1.shape,ftest_mean2.shape, relevance_ftest_mean2.shape)
                print(relevance_ftest_mean2.sum())
                relevance_ftest_mean1 = lrp_modules.compute_lrp_mean(ftest_mean2,ftest_mean1,relevance_ftest_mean2,dim=-1)
                print(relevance_ftest_mean1.sum())
                relevance_ftest_attended = lrp_modules.compute_lrp_mean(ftest_mean1, ftest_attended, relevance_ftest_mean1, dim=-1)
                print(relevance_ftest_attended.sum())
                relevance_ftest = relevance_ftest_attended
                # print(relevance_ftest_attended.shape)

                relevance = relevance_ftest.narrow(2, cls, 1).squeeze(2).view(batch_size*num_test, *f.size()[1:])
                print(relevance[0][2].sum())
                # with torch.enable_grad():
                relevance_images = lrp_wrapper.compute_lrp(base, xtest, target=relevance)

                relevance_images = relevance_images.view(batch_size, num_test, images_test.size(2), images_test.size(3), images_test.size(4))
                # print(relevance_images.sum())
                if args.nExemplars == 1:
                    relevance_ftrain_mean2 = relevance_ftest_mul_ftrain
                    # print(ftest_mean1.shape,ftest_mean2.shape, relevance_ftest_mean2.shape)
                    # print(relevance_ftest_mean2.sum())
                    relevance_train_mean1 = lrp_modules.compute_lrp_mean(ftrain_mean2, ftrain_mean1,
                                                                         relevance_ftrain_mean2, dim=-1)
                    # print(relevance_ftest_mean1.sum())
                    relevance_ftrain_attended = lrp_modules.compute_lrp_mean(ftrain_mean1, ftrain_attended,
                                                                            relevance_ftest_mean1, dim=-1)
                    # print(relevance_ftest_attended.sum())
                    relevance_ftrain = relevance_ftrain_attended

                for batch_size_idx in range(batch_size):
                    for img_idx in range(num_test):
                        image_path = images_test_pathes[img_idx][batch_size_idx]
                        # print(image_path)
                        gt_label = labels_test[batch_size_idx][img_idx]
                        predict_label = preds_index[batch_size_idx][img_idx]
                        gt_class = label_to_classlabel[int(gt_label.cpu().detach().numpy())]
                        predict_class = label_to_classlabel[int(predict_label.cpu().detach().numpy())]
                        img_filename = image_path.split('/')[-1]
                        img_relevance = relevance_images[batch_size_idx][img_idx]
                        print(img_relevance.sum(), gt_class, predict_class)
                        original_img = Image.fromarray(
                            np.uint8(project(images_test[batch_size_idx][img_idx].permute(1, 2, 0).cpu().numpy())))
                        if not os.path.isdir(os.path.join(save_folder, img_filename.strip('.jpg'))):
                            os.makedirs(os.path.join(save_folder, img_filename.strip('.jpg')))
                        if not os.path.exists(os.path.join(save_folder, img_filename.strip('.jpg'), gt_class + '_' + predict_class + img_filename)):
                            original_img.save(os.path.join(save_folder, img_filename.strip('.jpg'), gt_class + '_' + predict_class + img_filename))
                        hm = img_relevance.permute(1,2,0).unsqueeze(0).cpu().detach().numpy()
                        hm = LRPutil.gamma(hm)
                        hm = LRPutil.heatmap(hm)[0]
                        hm = project(hm)
                        hp_img = Image.fromarray(np.uint8(hm))
                        # plt.imshow(hm)
                        hp_img.save(os.path.join(save_folder, img_filename.strip('.jpg'), gt_class + '_' + label_to_classlabel[cls] + '_lrp_hm.jpg'))
                        # for one shot settings
                        if args.nExemplars == 1:
                            train_img = Image.fromarray(np.uint8(project(images_train[batch_size_idx][cls].permute(1, 2, 0).cpu().numpy())))

                            train_img.save(os.path.join(save_folder,img_filename.strip('.jpg'),  '_train_' + label_to_classlabel[int(labels_train[batch_size_idx][cls])] + '.jpg'))
                            test_attention = ftest_attention[batch_size_idx][cls][img_idx] # (1, m)
                            train_attention = ftrain_attention[batch_size_idx][cls][img_idx] # (1, m)
                            test_attention_heatmap = LRPutil.visuallize_attention(original_img,test_attention, reshape_size=(f_height, f_width))
                            train_attention_heatmap = LRPutil.visuallize_attention(train_img, train_attention, reshape_size=(f_height, f_width))
                            test_attention_heatmap.save(os.path.join(save_folder, img_filename.strip('.jpg'),
                                                                     gt_class + '_' + label_to_classlabel[int(labels_train[batch_size_idx][cls])] + '_test_attention_hm.jpg'))

                            train_attention_heatmap.save(os.path.join(save_folder, img_filename.strip('.jpg'),
                                                        gt_class + '_' + label_to_classlabel[int(labels_train[batch_size_idx][cls])] + '_train_attention_hm.jpg'))
                            relevance_train = relevance_ftrain[batch_size_idx][img_idx][cls:cls+1]
                            # print(relevance_train.shape)
                            xtrain_index = get_xtrain_idx(labels_train[batch_size_idx], cls)[0][0]
                            # print(xtrain_index)
                            relevance_train_img = lrp_wrapper.compute_lrp(base,xtrain[xtrain_index:xtrain_index+1], target=relevance_train).squeeze(0)
                            # print(relevance_train_img.shape)
                            hm = relevance_train_img.permute(1, 2, 0).unsqueeze(0).cpu().detach().numpy()
                            hm = LRPutil.gamma(hm)
                            hm = LRPutil.heatmap(hm)[0]
                            hm = project(hm)
                            hp_img = Image.fromarray(np.uint8(hm))
                            # plt.imshow(hm)
                            hp_img.save(os.path.join(save_folder, img_filename.strip('.jpg'),
                                                     gt_class + '_' + label_to_classlabel[cls] + '_train_lrp_hm.jpg'))




                        else:
                            test_attention = ftest_attention[batch_size_idx][cls][img_idx]  # (1, m)
                            test_attention_heatmap = LRPutil.visuallize_attention(original_img, test_attention,
                                                                                  reshape_size=(f_height, f_width))
                            test_attention_heatmap.save(os.path.join(save_folder, img_filename.strip('.jpg'),
                                                                     gt_class + '_' + label_to_classlabel[cls] + '_test_attention_hm.jpg'))



                # break
            # break