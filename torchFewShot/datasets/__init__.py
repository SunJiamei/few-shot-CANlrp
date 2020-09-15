from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .tieredImageNet import tieredImageNet
from .setdataset import Setdataset

__imgfewshot_factory = {
        'miniImagenet': Setdataset,
        'cub': Setdataset,
        'cars': Setdataset,
        'places': Setdataset,
        'omniglot': Setdataset,
        'emnist': Setdataset,
        'plantae': Setdataset,
        'tieredImageNet': tieredImageNet,
}

    # 'please specific dataset name using one of the options; [cars, cub, emnist, omniglot, places, plantae, tired_imagenet]'
def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](name,**kwargs)

