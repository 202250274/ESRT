import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import esrt #architecture,
from data import DIV2K, Set5_val, validation
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
from importlib import import_module
import pickle
from torchsummary import summary

# Loss 값을 불러옴
# with open('losses.pkl', 'rb') as f:
#     losses = pickle.load(f)

# # Loss 그래프 그리기
# plt.plot(losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss vs. Epochs')
# plt.show()

img1= np.load('result/mytest/synthetic/x2/0100x2.npy')
img_lr = np.load('dataset/DF2K_decoded/DIV2K_train_LR_bicubic/X2/0100.npy')
img_hr = np.load('dataset/DF2K_decoded/DIV2K_train_HR/0100x2.npy')
img1 = img1.reshape(256,-1).T
print(img_lr.shape)
img_lr = img_lr.reshape(128,-1).T
img_hr = img_hr.reshape(256,-1).T
# img1.tofile('result/mytest/set5/x2/0002x2.dat')
print(img1.shape)
print(img1.min())

plt.imshow(img_lr, cmap=plt.cm.gray)
plt.colorbar()
plt.figure()
plt.imshow(img_hr, cmap=plt.cm.gray)
plt.colorbar()
plt.figure()
plt.imshow(img1, cmap=plt.cm.gray)
plt.xlabel('x')
plt.ylabel('y')
plt.title('synthetic')
plt.colorbar()
plt.show()

# model = esrt.ESRT(upscale = 2, n_channels=1)
# summary(model,(1,128,128))