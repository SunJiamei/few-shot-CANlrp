from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader
import torch
import torchFewShot.transforms as T
import torchFewShot.datasets as datasets
import torchFewShot.dataset_loader as dataset_loader
from PIL import ImageEnhance
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
  def __init__(self, transformdict):
    self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

  def __call__(self, img):
    out = img
    randtensor = torch.rand(len(self.transforms))

    for i, (transformer, alpha) in enumerate(self.transforms):
      r = alpha*(randtensor[i]*2.0 -1.0) + 1
      out = transformer(out).enhance(r).convert('RGB')

    return out
class DataManager(object):
    """
    Few shot data manager
    """

    def __init__(self, args, use_gpu):
        super(DataManager, self).__init__()
        self.args = args
        self.use_gpu = use_gpu

        print("Initializing dataset {}".format(args.dataset))
        dataset = datasets.init_imgfewshot_dataset(name=args.dataset)

        if args.load:
            transform_train = T.Compose([
                T.RandomCrop(84, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(0.5)
            ])

            transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            transform_train = T.Compose([
                T.RandomResizedCrop(args.height),
                T.RandomHorizontalFlip(),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                # T.RandomVerticalFlip(),
                # T.RandomRotation(20),
                # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(0.5)
            ])

            if args.dataset != 'miniImageNet':
                transform_test = T.Compose([
                    T.Resize((int(args.height * 1.15), int(args.width * 1.15)), interpolation=3),
                    T.CenterCrop(args.height),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                transform_test = T.Compose([
                    T.Resize((int(args.height * 1.15), int(args.width * 1.15)), interpolation=3),
                    T.CenterCrop(args.height),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        pin_memory = True if use_gpu else False

        self.trainloader = DataLoader(
                dataset_loader.init_loader(name='train_loader',
                    dataset=dataset.train,
                    labels2inds=dataset.train_labels2inds,
                    labelIds=dataset.train_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.train_nTestNovel,
                    epoch_size=args.train_epoch_size,
                    transform=transform_train,
                    load=args.load,
                ),
                batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=True,
            )

        self.valloader = DataLoader(
                dataset_loader.init_loader(name='test_loader',
                    dataset=dataset.val,
                    labels2inds=dataset.val_labels2inds,
                    labelIds=dataset.val_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.nTestNovel,
                    epoch_size=args.epoch_size,
                    transform=transform_test,
                    load=args.load,
                ),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=False,
        )
        self.testloader = DataLoader(
                dataset_loader.init_loader(name='test_loader',
                    dataset=dataset.test,
                    labels2inds=dataset.test_labels2inds,
                    labelIds=dataset.test_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.nTestNovel,
                    epoch_size=args.epoch_size,
                    transform=transform_test,
                    load=args.load,
                ),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=False,
        )

    def return_dataloaders(self):
        if self.args.phase == 'test':
            return self.trainloader, self.testloader
        elif self.args.phase == 'val':
            return self.trainloader, self.valloader
