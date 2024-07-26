import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../../utils/")
sys.path.append("../")
from torch.utils.data import DataLoader
from utils import MyDataset, Normalize
from tqdm import tqdm
from networks.allaire_net import Allaire_Net
from networks.conv_net import Conv_Net
from networks.conv_net_2 import Conv_Net2
from networks.unet import U_Net
from networks.utils import EarlyStopper
import json
import argparse
import os

transform = Normalize(root="../../data")
dataset = MyDataset(root="../../data", tform=transform)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

net = U_Net({"cam": True})
net.load_state_dict(torch.load("../networks/weights/U-Net_50.pth"))

net.eval()

n = 0
for x, y in tqdm(train_loader):
    grad = []
    activation = []
    x_i, x_w = x
    pred = net((x_i, x_w))
    selected_action = pred >= 0.5
    loss = 0
    for i in range(400):
        for j in range(400):
            if selected_action[0, 0, i, j]:
                loss += pred[0,0,i,j]
    if loss:
        net.zero_grad()
        loss.backward()
        gradients = net.get_activations_gradient()
        grads = gradients.data.numpy().squeeze()
        activations = net.get_activations((x_i, x_w))
        fmap = activations.data.numpy().squeeze()
        tmp = grads.reshape(grads.shape[0], -1)
        weights = np.mean(tmp, axis = 1)
        cam = np.zeros(grads.shape[1:])
        for j,w in enumerate(weights):
            cam += w*fmap[j,:]
        cam = cam*(cam>0)
        heatmap = cam/(np.max(cam))
        heatmap = cv2.resize(heatmap, (400, 400))
        heatmap = np.array(heatmap, dtype='f')
        if y.sum() * 0.64 >= 100:
            plt.imsave(f'attention_maps/ewes/{n}.png', heatmap)
        else:
            plt.imsave(f'attention_maps/not_ewes/{n}.png', heatmap)
        n += 1