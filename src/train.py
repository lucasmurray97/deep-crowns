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
from networks.allaire_net import Allaire_Net

transform = Normalize()
dataset = MyDataset("../data", tform=transform)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, num_workers=8)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=8)

net = Allaire_Net()

epochs = 5
optimizer = torch.optim.Adam(net.parameters())
for epoch in tqdm(range(epochs)):
    n = 0
    for x, y in tqdm(train_loader):
        net.zero_grad()
        pred = net(x)
        loss = net.train_loss(pred, y)
        loss.backward()
        optimizer.step()
    for x, y in validation_loader:
        pred = net(x)
        loss = net.validation_loss(pred, y)
        n += 1
    net.reset_losses()
net.plot_loss(epochs=epochs)