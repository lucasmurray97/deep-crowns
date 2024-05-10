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
from networks.conv_net import Conv_Net
from networks.utils import EarlyStopper
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)

args = parser.parse_args()
epochs = args.epochs
lr = args.lr
wd = args.weight_decay
transform = Normalize()
dataset = MyDataset("../data", tform=transform)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

net = Conv_Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
net.cuda(0)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
early_stopper = EarlyStopper(patience=5, min_delta=0.01)
for epoch in tqdm(range(epochs)):
    n = 0
    for x, y in tqdm(train_loader):
        net.zero_grad()
        pred = net((x[0].cuda(0), x[1].cuda(0)))
        loss = net.train_loss(pred, y.cuda(0))
        loss.backward()
        optimizer.step()
    if early_stopper.early_stop(net.epoch_loss):             
      print("Early stoppage at epoch:", epoch)
      break
    """
    for x, y in validation_loader:
        pred = net(x)
        loss = net.validation_loss(pred, y)
        n += 1
    """
    net.reset_losses()
net.plot_loss(epochs=epochs)
net.finish(epochs)


accuracy = BinaryAccuracy()
precision = BinaryPrecision()
recall = BinaryRecall()
f1 = BinaryF1Score()
net.cuda(0)
net.eval()
for x, y in tqdm(train_loader):
    pred = net((x[0].cuda(0), x[1].cuda(0)))
    probs = pred.flatten()
    target = y.flatten().cuda(0)
    accuracy.update(probs, target.int())
    precision.update(probs, target.int())
    recall.update(probs, target.int())
    f1.update(probs, target.int())
results = {}
results["accuracy"] = accuracy.compute().item()
results["precision"] = precision.compute().item()
results["recall"] = recall.compute().item()
results["f1"] = f1.compute().item()
with open(f'./results_{net.name}_{epochs}.json', 'w') as f:
    json.dump(results, f)