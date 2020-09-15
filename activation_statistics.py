import sys
import argparse
from torchFewShot.datasets import setdataset
import torchFewShot.transforms as T
import torch
from torchFewShot.models.net import Model, ModelwithLRP
import os
import h5py
from sklearn.metrics.pairwise import euclidean_distances as pw_eudis
from PIL import Image
import torch
import numpy as np
import h5py
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('./torchFewShot')

'''extract and save the features of image encoder'''
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
def save_features(model, dataset, featurefile, transform):
    f = h5py.File(featurefile, 'w')
    max_count = len(dataset)
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (img_dir,label) in enumerate(dataset):
        if (i % 10) == 0:
            print('    {:d}/{:d}'.format(i, len(dataset)))
        x = read_image(img_dir)
        x = transform(x)
        x = x.cuda()  #(3, 84,84)
        feats = model.extract_feature(x)  #(1, 512, 6, 6)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = label
        count = count + feats.size(0)
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()
def extract_image_encoder_features():
    parser = argparse.ArgumentParser(description='Test image model with 5-way classification')
    # Datasets
    parser.add_argument('-d', '--dataset', type=str, default='miniImagenet')
    parser.add_argument('--load', default=False)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")
    # Optimization options
    parser.add_argument('--train-batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=1, type=int,
                        help="test batch size")
    # Architecture
    parser.add_argument('--num_classes', type=int, default=64)
    parser.add_argument('--scale_cls', type=int, default=7)
    parser.add_argument('--save-dir', type=str,
                        default='/home/sunjiamei/work/fewshotlearning/fewshot-CAN-master/result/miniImageNet/CAM/5-shot-seed1-resnet12-lrpscore-proto-test-224')
    parser.add_argument('--resume', type=str,
                        default='/home/sunjiamei/work/fewshotlearning/fewshot-CAN-master/result/miniImageNet/CAM/5-shot-seed1-resnet12-lrpscore-proto-test-224/best_model.pth.tar',
                        metavar='PATH')
    # FewShot settting
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=5,
                        help='number of training examples per novel category.')
    parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of episodes per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=16 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')
    # Miscs
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu-devices', default='0', type=str)
    args = parser.parse_args()
    testdataset = setdataset.Setdataset(args.dataset)
    testset = testdataset.test
    model = Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    # load the model
    model.cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    transform_test = T.Compose([
        T.Resize((int(args.height * 1.15), int(args.width * 1.15)), interpolation=3),
        T.CenterCrop(args.height),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feature_file = os.path.join(args.save_dir, 'features', args.dataset + '.hdf5')
    if os.path.exists(feature_file):
        print('the feature file exists, we will return none')
    else:
        # extract features
        print(f'==extracting features of {args.dataset} with model {args.resume}')
        save_features(model, testset, feature_file, transform_test)
        print('feature_saved to:', feature_file)


'''read the saved features'''
class SimpleHDF5Dataset:
  def __init__(self, file_handle = None):
    if file_handle == None:
      self.f = ''
      self.all_feats_dset = []
      self.all_labels = []
      self.total = 0
    else:
      self.f = file_handle
      self.all_feats_dset = self.f['all_feats'][...]
      self.all_labels = self.f['all_labels'][...]
      self.total = self.f['count'][0]
  def __getitem__(self, i):
    return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

  def __len__(self):
    return self.total

def init_loader(filename):
  with h5py.File(filename, 'r') as f:
    fileset = SimpleHDF5Dataset(f)

  feats = fileset.all_feats_dset
  labels = fileset.all_labels
  while np.sum(feats[-1]) == 0:
    feats  = np.delete(feats,-1,axis = 0)
    labels = np.delete(labels,-1,axis = 0)

  class_list = np.unique(np.array(labels)).tolist()
  inds = range(len(labels))

  cl_data_file = {}
  for cl in class_list:
    cl_data_file[cl] = []
  for ind in inds:
    cl_data_file[labels[ind]].append( feats[ind])

  return cl_data_file


def analyze_statistics(feature_file_name, save_path, pooling_quantile):
    cl_in = init_loader(feature_file_name)
    '''calculate the statistics of the feature dimension, mean, quantile, variance, median'''
    statistics = {}
    var = []
    quantile = []
    median = []
    mean = []
    pos_num = []
    max_value = []
    quantile_diff1 = []
    quantile_diff2 = []
    quantile_diff3 = []
    quantile_diff4 = []
    count = 0
    # quantile_points = [0.3, 0.5, 0.9]
    quantile_points = np.arange(20) * 5 / 100
    quantile_points = quantile_points[1:]
    for key in cl_in.keys():
        for feat in cl_in[key]:
            # print(feat.shape)
            pos_num.append(np.sum(feat > 0))
            feat = feat.reshape(512, -1)
            if pooling_quantile == 'mean':
                feat = np.mean(feat, axis=-1)
            else:
                feat = np.quantile(feat, [pooling_quantile], axis=-1)
            # feat = feat.median([-1,-2])
            # print(feat.shape)
            max_value.append(np.max(feat))
            var.append(np.var(feat))
            mean.append(np.mean(feat))
            median.append(np.median(feat))
            quantile_value = np.quantile(feat, quantile_points)
            # print(len(quantile_value))
            quantile.append(quantile_value)
            quantile_diff1.append(quantile_value[18] - quantile_value[8])
            quantile_diff2.append(quantile_value[16] - quantile_value[6])
            quantile_diff3.append(quantile_value[14] - quantile_value[4])
            quantile_diff4.append(quantile_value[12] - quantile_value[2])
            # print(var, mean, median, quantile)
        #     count += 1
        #     if count >=3:
        #         break
        # break
    mean = np.array(mean)
    var = np.array(var)
    median = np.array(median)
    quantile = np.array(quantile)
    pos_num = np.array(pos_num)
    max_value = np.array(max_value)
    quantile_diff1 = np.array(quantile_diff1)
    quantile_diff2 = np.array(quantile_diff2)
    quantile_diff3 = np.array(quantile_diff3)
    quantile_diff4 = np.array(quantile_diff4)
    print(len(mean), len(var), len(median), len(quantile))
    statistics['mean'] = [np.mean(mean), np.std(mean)]
    statistics['var'] = [np.mean(var), np.std(var)]
    statistics['median'] = [np.mean(median), np.std(median)]
    statistics['pos_num'] = [np.mean(pos_num), np.std(pos_num)]
    statistics['maximum'] = [np.mean(max_value), np.std(max_value)]
    statistics['quantile_diff1'] = [np.mean(quantile_diff1), np.std(quantile_diff1)]
    statistics['quantile_diff2'] = [np.mean(quantile_diff2), np.std(quantile_diff2)]
    statistics['quantile_diff3'] = [np.mean(quantile_diff3), np.std(quantile_diff3)]
    statistics['quantile_diff4'] = [np.mean(quantile_diff4), np.std(quantile_diff4)]
    statistics['quantile'] = {}
    for i, p in enumerate(quantile_points):
        statistics['quantile'][p] = [np.mean(quantile[:, i]), np.std(quantile[:, i])]
    print(statistics)
    with open(save_path, 'w') as fs:
        for key in statistics.keys():
            if key != 'quantile':
                fs.write(key + ': ' + str(statistics[key][0]) + '+-' + str(statistics[key][1]) + '\n')
            else:
                for subkey in statistics['quantile']:
                    fs.write('quantile' + str(subkey) + ': '+ str(statistics['quantile'][subkey][0]) + '+-' + str(statistics['quantile'][subkey][1]) + '\n')
        fs.close()
    return statistics
def draw_scatter_points(x,y, savepath):
    # print(x.shape,y.shape)
    matplotlib.rcParams.update({'font.size': 18})
    color_map = ['b', 'g', 'r', 'y', 'c']
    # create a scatter plot.
    # f = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(aspect='equal')
    plt.figure(figsize=(10,10))
    # plt.axis([-25, 25, -25, 25])
    # axis = [-50,50,-50,50]
    axis = [-25, 25, -25, 25]
    plt.scatter(axis[0], axis[2],s=1, c='w')
    plt.scatter(axis[0], axis[3],s=1, c='w')
    plt.scatter(axis[1], axis[2],s=1, c='w')
    plt.scatter(axis[1], axis[3],s=1, c='w')
    for i in range(len(x)):
        if x[i,0]<axis[0] or x[i,0]>axis[1]:
            continue
        if x[i,1]<axis[2] or x[i,1]>axis[3]:
            continue
        plt.axis(axis)
        plt.scatter(x[i, 0], x[i, 1],s=5, c=color_map[int(y[i]-1)])
        plt.axis(axis)
    plt.axis('tight')
    plt.savefig(savepath,bbox_inches='tight')
    # plt.show()
def cal_distance(features):
    dis = pw_eudis(features, features)
    return dis
def tsne_visualization(feature_file_path):
    label_dict = {'cars':2, 'cub':3, 'places':4, 'plantae':5, 'miniImagenet':1}
    features = []
    labels = []
    num = 1000
    for set in ['cars', 'cub', 'places', 'plantae', 'miniImagenet']:
        feature_file_name = os.path.join(feature_file_path, set + ".hdf5")
        cl_in = init_loader(feature_file_name)
        local_feature = []
        for key in cl_in.keys():
            for feat in cl_in[key]:
                feat = feat.reshape(512, -1)
                # print(feat.shape)
                feat = np.quantile(feat, [0.95], axis=-1)[0]
                local_feature.append(feat)
        random.shuffle(local_feature)
        features += local_feature[:num]
        labels += [label_dict[set]]*num
    features = np.array(features)
    labels = np.array(labels)
    print(features.shape)
    features_tsne = TSNE(random_state=0).fit_transform(features)
    # features_tsne = features[:, :2]
    pw_dis = cal_distance(features_tsne)
    mean_pairwise_dis = np.mean(pw_dis)
    print(np.mean(pw_dis), np.std(pw_dis))
    print(features_tsne.shape)
    draw_scatter_points(features_tsne,labels,os.path.join(feature_file_path, 'canlrpminiImagenet_0.95.jpg'))


def tsne_visualization_sepdomains(feature_file_path):
    label_dict = {'miniImagenet':1,'cars':2, 'cub':3, 'places':4, 'plantae':5}
    num = 2000
    base_feature_file_name = os.path.join(feature_file_path, "miniImagenet.hdf5")
    base_cl_in = init_loader(base_feature_file_name)
    local_feature = []
    for key in base_cl_in.keys():
        for feat in base_cl_in[key]:
            feat = feat.reshape(512, -1)
            # print(feat.shape)
            # feat = feat.mean(-1)
            feat = np.quantile(feat, [0.9], axis=-1)[0]
            local_feature.append(feat)
    random.shuffle(local_feature)
    base_features = local_feature[:num]
    base_labels = [label_dict['miniImagenet']]*num
    for set in ['cars', 'cub', 'places', 'plantae']:
        feature_file_name = os.path.join(feature_file_path, set + ".hdf5")
        cl_in = init_loader(feature_file_name)
        local_feature = []
        for key in cl_in.keys():
            for feat in cl_in[key]:
                feat = feat.reshape(512, -1)
                # print(feat.shape)
                # feat = feat.mean(-1)
                feat = np.quantile(feat, [0.9], axis=-1)[0]
                local_feature.append(feat)
        random.shuffle(local_feature)
        features = base_features + local_feature[:num]
        labels = base_labels + [label_dict[set]]*num
        features = np.array(features)
        labels = np.array(labels)
        print(features.shape)
        features_tsne = TSNE(random_state=0).fit_transform(features)
        # features_tsne = features[:, :2]
        print(features_tsne.shape)
        draw_scatter_points(features_tsne, labels, os.path.join(feature_file_path, 'canlrpminiImagenet_'+set+'.jpg'))



def read_statistics(file_path):
    statistics = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key = line.split(':')[0]
        sta = line.split(':')[-1]
        sta_mean = sta.split('+-')[0]
        sta_std = sta.split('+-')[1]
        sta_mean = float(sta_mean)
        sta_std = float(sta_std)
        statistics[key] = [sta_mean, sta_std]
    return statistics


def plot_statistics():
    path_lrp = './result/miniImageNet/CAM/5-shot-seed1-resnet12-lrpscore-proto-test-224/features'
    path = './result/miniImageNet/CAM/5-shot-seed1-resnet12-224/features'
    save_path = './result/miniImageNet/CAM/'
    bar_width = 0.2
    for pool in ['meanpooling', '0.9quantilepooling','0.95quantilepooling']:
        #=========plot var====================
        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['var'][0],  yerr=sta['var'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['var'][0], yerr=stalrp['var'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['var'][0],  yerr=sta['var'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['var'][0], yerr=stalrp['var'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.show()
        plt.ylabel('variance', fontsize=20)
        plt.savefig(os.path.join(save_path, f'{pool}_variance.jpg'), bbox_inches='tight')
        #=============plot 95-45================
        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['quantile_diff1'][0],  yerr=sta['quantile_diff1'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile_diff1'][0], yerr=stalrp['quantile_diff1'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['quantile_diff1'][0],  yerr=sta['quantile_diff1'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile_diff1'][0], yerr=stalrp['quantile_diff1'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        plt.ylabel('95%quantile-45%quantile', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_quantile_diff95-45.jpg'), bbox_inches='tight')
        #=============plot mean ================
        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['mean'][0],  yerr=sta['mean'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['mean'][0], yerr=stalrp['mean'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['mean'][0],  yerr=sta['mean'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['mean'][0], yerr=stalrp['mean'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.show()
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('mean', fontsize=20)
        plt.savefig(os.path.join(save_path, f'{pool}_mean.jpg'), bbox_inches='tight')
        #=============plot pos num =============
        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['pos_num'][0],  yerr=sta['pos_num'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['pos_num'][0], yerr=stalrp['pos_num'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['pos_num'][0],  yerr=sta['pos_num'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['pos_num'][0], yerr=stalrp['pos_num'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.show()
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('#positive activation', fontsize=20)
        plt.savefig(os.path.join(save_path, f'{pool}_pos_num.jpg'), bbox_inches='tight')
        #=============plot median===============
        plt.figure()
        plt.xticks(np.arange(4) + 1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['median'][0], yerr=sta['median'][1], alpha=0.6, width=bar_width,
                        facecolor='darkblue', edgecolor='white', label='can', lw=1)
                plt.bar(x_pos + bar_width, stalrp['median'][0], yerr=stalrp['median'][1], alpha=0.6, width=bar_width,
                        facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['median'][0], yerr=sta['median'][1], alpha=0.6, width=bar_width,
                        facecolor='darkblue', edgecolor='white', lw=1)
                plt.bar(x_pos + bar_width, stalrp['median'][0], yerr=stalrp['median'][1], alpha=0.6, width=bar_width,
                        facecolor='deeppink', edgecolor='white', lw=1)
            x_pos += 1
            # plt.legend(loc=2)
        plt.axis('tight')
        # plt.show()
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('median', fontsize=20)
        plt.savefig(os.path.join(save_path, f'{pool}_median.jpg'), bbox_inches='tight')
        #===============plot first order statistics variance across the images =================
        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['mean'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['mean'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['mean'][1],  alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['mean'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('mean', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_mean_variance.jpg'), bbox_inches='tight')
        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['median'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['median'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['median'][1],  alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['median'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('median', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_median_variance.jpg'), bbox_inches='tight')

        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['quantile0.75'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.75'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['quantile0.75'][1],  alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.75'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('quantile0.75', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_quantile0.75_variance.jpg'), bbox_inches='tight')

        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['quantile0.45'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.45'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['quantile0.45'][1],  alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.45'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('quantile0.45', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_quantile0.45_variance.jpg'), bbox_inches='tight')

        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['quantile0.95'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.95'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['quantile0.95'][1],  alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.95'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('quantile0.95', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_quantile0.95_variance.jpg'), bbox_inches='tight')

        plt.figure()
        plt.xticks(np.arange(4)+1, ('cars', 'cub', 'places', 'plantae'), fontsize=24)
        x_pos = 1
        for setname in ['cars', 'cub', 'places', 'plantae']:
            stalrp_path = os.path.join(path_lrp, f'canlrp_{setname}_statistics_{pool}.txt')
            sta_path = os.path.join(path, f'can_{setname}_statistics_{pool}.txt')
            stalrp = read_statistics(stalrp_path)
            sta = read_statistics(sta_path)
            # print(sta)
            if setname == 'cars':
                plt.bar(x_pos, sta['quantile0.35'][1], alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', label='can', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.35'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white', label='canlrp', lw=1)
            else:
                plt.bar(x_pos, sta['quantile0.35'][1],  alpha=0.6, width = bar_width, facecolor = 'darkblue', edgecolor = 'white', lw=1)
                plt.bar(x_pos+bar_width, stalrp['quantile0.35'][1], alpha=0.6, width = bar_width, facecolor='deeppink', edgecolor='white',  lw=1)
            x_pos +=1
            #plt.legend(loc=2)
        plt.axis('tight')
        # plt.xlabel('datasets', fontsize=20)
        plt.ylabel('quantile0.35', fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(save_path, f'{pool}_quantile0.35_variance.jpg'), bbox_inches='tight')


if __name__ == '__main__':
    plot_statistics()
    # extract_image_encoder_features()
    # feature_file_path = './result/miniImageNet/CAM/5-shot-seed1-resnet12-lrpscore-proto-test-224/features'
    # feature_file_path = './result/miniImageNet/CAM/5-shot-seed1-resnet12-224/features'
    # tsne_visualization(feature_file_path)
    # tsne_visualization_sepdomains(feature_file_path)
    # for path in [ './result/miniImageNet/CAM/5-shot-seed1-resnet12-lrpscore-proto-test-224/features','./result/miniImageNet/CAM/5-shot-seed1-resnet12-224/features' ]:
    #     if 'lrp' in path:
    #         name = 'canlrp'
    #     else:
    #         name = 'can'
    #     for set in ['cars', 'cub', 'places', 'plantae']:
    #         feature_file_name = os.path.join( path, set + ".hdf5")
    #         statistics_file_name = os.path.join(path, name + '_' + set + f'_statistics_meanpooling.txt')
    #         analyze_statistics(feature_file_name, statistics_file_name, 'mean')
    #         for quantile in [0.9,0.95]:
    #             statistics_file_name = os.path.join(path, name + '_' + set + f'_statistics_{quantile}quantilepooling.txt')
    #             analyze_statistics(feature_file_name, statistics_file_name, quantile)
