import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import HTML, display
import os
from keras.utils.visualize_util import plot

from deepOs.statefarm import Statefarm
from deepOs.resnet import Resnet

# create instance of statefarm helper class:
use_cache = 1
path_data = "/work/hofinger/code/python/statefarm/input" #"/net/store/ni/projects/deeplearning/statefarm"
path_cache = "/work/hofinger/code/python/statefarm/cache" #"/net/store/ni/projects/deeplearning/statefarm/cache"
statefarm = Statefarm(use_cache, path_data, path_cache)

# create network
resnet = Resnet(statefarm)

np.random.seed(1234)
train_driver_ids = np.arange(0,23)
valid_driver_ids = np.arange(23,26)

# train network
history, predictions_valid, log_loss_valid_per_sample = resnet.train(train_driver_ids, valid_driver_ids)


