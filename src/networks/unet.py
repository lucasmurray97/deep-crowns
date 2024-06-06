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

class U_Net(nn.Module):
    def __init__(self, params = {}):
        super(U_Net, self).__init__()
        self.name = "U-Net"
        self.base_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=9, out_channels=1, init_features=32)
        self.fcw_1 = nn.Linear(in_features=2, out_features=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.upconv_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4,4), stride=(7,7))
        self.conv1 = nn.Conv2d(in_channels=513, out_channels=512, kernel_size=(1,1), stride=(1,1))
        self.bn2 = nn.BatchNorm2d(512)
        # Loss func
        self.loss = []
        self.epoch_loss = 0
        self.val_loss = []
        self.val_epoch_loss = 0
        self.n = 0
        self.m = 0

    def forward(self, x):
        x_i, x_w = x
        # Matrix processing
        enc1 = self.base_model.encoder1(x_i.float())
        #print(enc1.shape)
        enc2 = self.base_model.encoder2(self.base_model.pool1(enc1))
        #print(enc2.shape)
        enc3 = self.base_model.encoder3(self.base_model.pool2(enc2))
        #print(enc3.shape)
        enc4 = self.base_model.encoder4(self.base_model.pool3(enc3))
        #print(enc4.shape)
        bottleneck = self.base_model.bottleneck(self.base_model.pool4(enc4))
        # Vector and matrix combination:
        x_w = F.relu(self.bn1(self.fcw_1(x_w)))
        x_w = x_w.view(-1, 1, 4, 4)
        #print(x_w.shape)
        x_w = self.upconv_1(x_w)
        #print(x_w.shape)
        combination = torch.cat((bottleneck, x_w), dim = 1)
        #print(combination.shape)
        x_comb = F.relu(self.bn2(self.conv1(combination)))
        #print(x_comb.shape)
        dec4 = self.base_model.upconv4(x_comb)
        #print(dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        #print(dec4.shape)
        dec4 = self.base_model.decoder4(dec4)
        #print(dec4.shape)
        dec3 = self.base_model.upconv3(dec4)
        #print(dec3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        #print(dec3.shape)
        dec3 = self.base_model.decoder3(dec3)
        #print(dec3.shape)
        dec2 = self.base_model.upconv2(dec3)
        #print(dec2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        #print(dec2.shape)
        dec2 = self.base_model.decoder2(dec2)
        #print(dec2.shape)
        dec1 = self.base_model.upconv1(dec2)
        #print(dec1.shape)
        dec1 = torch.cat((dec1, enc1), dim=1)
        #print(dec1.shape)
        dec1 = self.base_model.decoder1(dec1)
        #print(dec1.shape)
        out = x = F.sigmoid(self.base_model.conv(dec1))
        #print(out.shape)
        return out

    def train_loss(self, x, y):
        loss = F.binary_cross_entropy(x.view(-1, 400*2), y.view(-1, 400*2), reduction='sum')
        self.epoch_loss += loss.item()
        self.n += 1
        return loss
    
    def validation_loss(self, x, y):
        val_loss = F.binary_cross_entropy(x.view(-1, 400*2), y.view(-1, 400*2), reduction='sum')
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

    def finish(self, epochs):
        self.plot_loss(epochs)
        path_ = f"./networks/weights/{self.name}_{epochs}.pth"
        torch.save(self.state_dict(), path_)