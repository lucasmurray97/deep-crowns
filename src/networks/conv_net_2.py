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

class Conv_Net2(nn.Module):
    def __init__(self, params = {}):
        super(Conv_Net2, self).__init__()
        self.name = "Conv-Net2"
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=(3,3), stride=(1,1)) # (64, 10, 10)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1)) # (64, 10, 10)
        self.bn2 = nn.BatchNorm2d(128)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1)) # (64, 10, 10)
        self.bn3 = nn.BatchNorm2d(256)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1)) # (64, 10, 10)
        self.bn4_ = nn.BatchNorm2d(256)
        self.max_pool_4_= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(3,3), stride=(1,1)) # (64, 10, 10)
        self.bn5_ = nn.BatchNorm2d(1024)
        self.max_pool_5_= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(3,3), stride=(1,1)) # (64, 10, 10)
        self.bn6_ = nn.BatchNorm2d(2048)
        self.max_pool_6_= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fcw_1 = nn.Linear(in_features=2, out_features=16)
        self.bn6 = nn.BatchNorm1d(16)
        self.conv1_ = nn.ConvTranspose2d(in_channels=2049, out_channels=1024, kernel_size=(4,4), stride=(2,2))
        self.bn9 = nn.BatchNorm2d(1024)
        self.conv2_ = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(5,5), stride=(2,2))
        self.bn10 = nn.BatchNorm2d(512)
        self.conv3_ = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4), stride=(2,2))
        self.bn11 = nn.BatchNorm2d(256)
        self.conv4_ = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=(2,2))
        self.bn12 = nn.BatchNorm2d(128)
        self.conv5_ = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=(2,2))
        self.bn13 = nn.BatchNorm2d(64)
        self.conv6_ = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(4,4), stride=(2,2))
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
        x_i = self.max_pool_4_(F.relu(self.bn4_(self.conv4(x_i))))
        #print(x_i.shape)
        x_i = self.max_pool_5_(F.relu(self.bn5_(self.conv5(x_i))))
        #print(x_i.shape)
        x_i = self.max_pool_6_(F.relu(self.bn6_(self.conv6(x_i))))
        #print(x_i.shape)
        x_i = x_i.flatten(start_dim = 1)
        x_w = F.relu(self.bn6(self.fcw_1(x_w)))
        x = torch.cat([x_i, x_w], dim=1)
        #print(x.shape)
        x = x.view(x.size(0), 2049, 4, 4)
        #print(x.shape)
        x = F.relu(self.bn9(self.conv1_(x)))
        #print(x.shape)
        x = F.relu(self.bn10(self.conv2_(x)))
        #print(x.shape)
        x = F.relu(self.bn11(self.conv3_(x)))
        #print(x.shape)
        x = F.relu(self.bn12(self.conv4_(x)))
        #print(x.shape)
        x = F.relu(self.bn13(self.conv5_(x)))
        #print(x.shape)
        x = F.sigmoid(self.conv6_(x))
        #print(x.shape)
        return x
        
    def train_loss(self, x, y):
        loss = F.binary_cross_entropy(x.view(-1, 32000), y.view(-1, 32000), reduction='sum')
        self.epoch_loss += loss.item()
        self.n += 1
        return loss
    
    def validation_loss(self, x, y):
        val_loss = F.binary_cross_entropy(x.view(-1, 32000), y.view(-1, 32000), reduction='sum')
        self.val_epoch_loss += val_loss.item()
        self.m += 1
        return val_loss
    
    def reset_losses(self):
        self.loss.append(self.epoch_loss/self.n)
        self.val_loss.append(self.val_epoch_loss/self.m)
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
        plt.savefig(f"./plots/Losses_{self.name}_{epochs}.png")

    def finish(self, epochs):
        self.plot_loss(epochs)
        path_ = f"./networks/weights/{self.name}_{epochs}.pth"
        torch.save(self.state_dict(), path_)


    
