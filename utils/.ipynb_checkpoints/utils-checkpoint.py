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

class MyDataset(torch.utils.data.Dataset):
    """Creates dataset that sampes: (landscape + fire_t, isocrone_(t+1)).
    landscape = (fuels, arqueo, canopy bulk density, canopy base height, elevation, flora, paleo, urban)
    fire = bit mask of fire current state
    isocrone = bit mask of fire evolution at time t+1
    Args:
        root (str): directory where data is being stored
        tform (Transform): tranformation to be applied at sampling time.
    """
    def __init__(self, root, tform=None):
        super(MyDataset, self).__init__()
        self.root = root
        HOME_FOLDER = pathlib.Path(root + "/spreads/")
        self.fires = []
        self.isoc = []
        self.n = 0
        dir_list = set(list(i.name.split('_')[0].split('-')[0] for i in HOME_FOLDER.iterdir()))
        for item in HOME_FOLDER.iterdir():
            if "fire" in item.name:
                self.fires.append(item.name)
            else:
                self.isoc.append(item.name)
        self.n = len(self.fires) - len(dir_list)
        self.transform = tform
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        potential = self.fires[i]
        n = None
        m = None
        if potential.split("_")[1].split("-")[0] == self.fires[i+1].split("_")[1].split("-")[0]:
            if potential.split("_")[1].split("-")[1].split(".")[0] == "0":
                n,m = self.fires[i+1], self.isoc[i+2]
            else:
                n,m =  self.fires[i], self.isoc[i+1]
        else:
            n,m =  self.fires[i+1], self.isoc[i+2]
        fire_number = n.split('_')[1].split('-')[0]
        spread_number = n.split('_')[1].split('.png')[0]
        iso_number = m.split('_')[1].split('.png')[0]
        with fiona.open(f"{self.root}/shapes/box_{fire_number}.shp", "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        with rioxarray.open_rasterio(f"{self.root}/landscape/Input_Geotiff.tif") as src:
            out_image = src.rio.clip(shapes).values
            out_image = np.where(out_image == -9999.0, -1, out_image)
        spread = read_image(f"{self.root}/spreads/fire_{spread_number}.png")
        spread = torch.where(spread[1] == 231, 1.0, 0.0)
        isoc = read_image(f"{self.root}/spreads/iso_{iso_number}.png")
        isoc = torch.where(isoc[1] == 231, 1.0, 0.0).unsqueeze(0)
        input = torch.cat((spread.unsqueeze(0), torch.from_numpy(out_image)))
        if self.transform:
            input = self.transform(input)
            isoc = self.transform(isoc)
        return input, isoc
            

        

class Normalize(object):
    """Normalizes image channnel by channel.

    Args:
        image (Tensor): image of dimension (C x H x W), where normalization will be carried out
        independently for the C channels
    """

    def __init__(self, root ="../data"):
        self.root = root
        with rioxarray.open_rasterio(f"{self.root}/landscape/Input_Geotiff.tif") as src:
            self.maxes = src.max(dim=["x", "y"]).values
        

    def __call__(self, image):
        shape = image.shape
        if shape[0] > 1:
            for i in range(1, shape[0]):
                image[i].apply_(lambda x: (x + 1)/(self.maxes[i-1] + 1))
            return image
        else:
            return image
    