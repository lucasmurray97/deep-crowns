import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../utils/")
from torch.utils.data import DataLoader
from utils import MyDataset, Normalize
from tqdm import tqdm

class Conv_Net(nn.Module):
    def __init__(self, params = {}):
        super(Conv_Net, self).__init__()
        self.name = "Conv-Net"
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3,3), stride=(2,2)) # (64, 10, 10)
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool_1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2)) # (64, 10, 10)
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool_2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(1,1)) # (64, 10, 10)
        self.bn3 = nn.BatchNorm2d(32)
        self.max_pool_3 = nn.MaxPool2d(2)
        self.fci_1 = nn.Linear(in_features=2816, out_features=1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fcw_1 = nn.Linear(in_features=2, out_features=64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc_1 = nn.Linear(in_features=1088, out_features=1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.up1 = nn.Upsample(scale_factor = 4)
        self.conv1_ = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,2))
        self.up2 = nn.Upsample(size=(298, 390))
        self.conv2_ = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,1))
        self.loss = []
        self.epoch_loss = 0
        self.val_loss = []
        self.val_epoch_loss = 0
        self.n = 0
        self.m = 0
        
    def forward(self, x):
        x_i, x_w = x
        #print(x_i.shape)
        x_i = self.max_pool_1(F.relu(self.bn1(self.conv1(x_i.float()))))
        #print(x_i.shape)
        x_i = self.max_pool_2(F.relu(self.bn2(self.conv2(x_i))))
        #print(x_i.shape)
        x_i = self.max_pool_3(F.relu(self.bn3(self.conv3(x_i))))
        #print(x_i.shape)
        x_i = x_i.flatten(start_dim = 1)
        #print(x_i.shape)
        x_i = F.relu(self.bn5(self.fci_1(x_i)))
        #print(x_i.shape)
        x_w = F.relu(self.bn6(self.fcw_1(x_w)))
        #print(x_w.shape)
        x = torch.cat([x_i, x_w], dim=1)
        #print(x.shape)
        x = F.relu(self.bn7(self.fc_1(x)))
        #print(x.shape)
        x = x.view(x.size(0), 1, 32, 32)
        #print(x.shape)
        x = F.relu(self.conv1_(self.up1(x)))
        #print(x.shape)
        x = F.sigmoid(self.conv2_(self.up2(x)))
        #print(x.shape)
        return x
        
    def train_loss(self, x, y):
        loss = F.binary_cross_entropy(x.view(-1, 116220), y.view(-1, 116220), reduction='sum')
        self.epoch_loss += loss.item()
        self.n += 1
        return loss
    
    def validation_loss(self, x, y):
        val_loss = F.binary_cross_entropy(x.view(-1, 116220), y.view(-1, 116220), reduction='sum')
        self.val_epoch_loss += val_loss.item()
        self.m += 1
        return val_loss
    
    def reset_losses(self):
        self.loss.append(self.epoch_loss/self.n)
        #self.val_loss.append(self.val_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.n = 0
        self.m = 0
        
    def plot_loss(self, epochs):
        self.to("cpu")
        plt.ion()
        fig = plt.figure()
        plt.plot(self.loss, label='training loss')
        plt.plot(self.val_loss, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"Losses_{self.name}_{epochs}.png")

    
