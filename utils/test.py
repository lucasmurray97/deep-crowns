import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt
from torchvision.io import read_image
import fiona
import rasterio
import rasterio.mask
import pathlib
import rioxarray
from utils import MyDataset, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

transform = Normalize()
dataset = MyDataset("../data", tform=transform)