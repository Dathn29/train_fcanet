import torch
from torchvision import transforms as tr
from torch.utils.data import DataLoader

import os
import cv2
import time
import math
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from scipy.ndimage.morphology import distance_transform_edt

import helper
import transform as tf
from dataset import GeneralDataset
import matplotlib.pyplot as plt

transform_train = tr.Compose([
                            tf.RGBShift(0, 0, 100),
                            tf.MatchShortSideResize(size=512),
                            tf.ITIS_Crop(itis_pro=0.7, mode='strategy#05',crop_size=(512,512)),
                            tf.CatPointMask(mode='DISTANCE_POINT_MASK_SRC'),
                            tf.CatPointMask(mode='DISTANCE_POINT_MASK_FIRST'),
                             ])
train_set = GeneralDataset("data/PASCAL_SBD/list/val.txt", transform_train)
print(train_set[0]['img'].shape)
plt.imshow(train_set[0]['img'])
plt.show()