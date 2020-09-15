from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

# class miniImageNet(object):
#     """
#     Dataset statistics:
#     # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
#     """
#     dataset_dir = '/home/sunjiamei/work/fewshotlearning/dataset/miniImageNet/'
#
#     def __init__(self):
#         super(miniImageNet, self).__init__()
#         self.train_dir = os.path.join(self.dataset_dir, 'train')
#         self.val_dir = os.path.join(self.dataset_dir, 'val')
#         self.test_dir = os.path.join(self.dataset_dir, 'test')
#
#         self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
#         self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
#         self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)
#
#         self.num_train_cats = len(self.train_labelIds)
#         num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
#         num_total_imgs = len(self.train + self.val + self.test)
#
#         print("=> MiniImageNet loaded")
#         print("Dataset statistics:")
#         print("  ------------------------------")
#         print("  subset   | # cats | # images")
#         print("  ------------------------------")
#         print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
#         print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
#         print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
#         print("  ------------------------------")
#         print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
#         print("  ------------------------------")
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.val_dir):
#             raise RuntimeError("'{}' is not available".format(self.val_dir))
#         if not osp.exists(self.test_dir):
#             raise RuntimeError("'{}' is not available".format(self.test_dir))
#
#     def _process_dir(self, dir_path):
#         cat_container = sorted(os.listdir(dir_path))
#         cats2label = {cat:label for label, cat in enumerate(cat_container)}
#
#         dataset = []
#         labels = []
#         for cat in cat_container:
#             for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):
#                 if '.jpg' not in img_path:
#                     continue
#                 label = cats2label[cat]
#                 dataset.append((os.path.join(dir_path, cat, img_path), label))
#                 labels.append(label)
#
#         labels2inds = {}
#         for idx, label in enumerate(labels):
#             if label not in labels2inds:
#                 labels2inds[label] = []
#             labels2inds[label].append(idx)
#
#         labelIds = sorted(labels2inds.keys())
#         return dataset, labels2inds, labelIds
class miniImageNet(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """
    dataset_dir = '/home/sunjiamei/work/fewshotlearning/dataset/miniImagenet/'

    def __init__(self):
        super(miniImageNet, self).__init__()
        self.train_dir = os.path.join(self.dataset_dir, 'base.json')
        self.val_dir = os.path.join(self.dataset_dir, 'val.json')
        self.test_dir = os.path.join(self.dataset_dir, 'novel.json')

        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> MiniImageNet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.isfile(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.isfile(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not os.path.isfile(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
    def _process_dir(self, dir_path):
        data = json.load(open(dir_path, 'r')) # ['label_names', 'image_names', 'image_labels']
        dataset_img_path = data['image_names']  # image_path
        labels = data['image_labels']  # the label of the images
        assert len(dataset_img_path) == len(labels)
        dataset = []
        for i in range(len(labels)):
            dataset.append((dataset_img_path[i], labels[i])) # label is the index of the category
        labels2inds = {}
        for idx, label in enumerate(labels):
            if label not in labels2inds:
                labels2inds[label] = []
            labels2inds[label].append(idx)
        labelIds = sorted(labels2inds.keys())
        # print(len(dataset),len(labels2inds),len(labelIds))
        # print(labels2inds)
        for item in dataset_img_path:
            assert '/dataset/' in item
        return dataset, labels2inds, labelIds