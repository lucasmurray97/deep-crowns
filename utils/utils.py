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
import pandas as pd
import os
from tqdm import tqdm

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
        self.data = {}
        
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        potential = self.fires[i]
        n = None
        m = None
        if potential.split("_")[1].split("-")[0] == self.fires[i+1].split("_")[1].split("-")[0]:
                n,m =  self.fires[i], self.isoc[i+1]
        else:
            n,m =  self.fires[i+1], self.isoc[i+2]
        fire_number = n.split('_')[1].split('-')[0]
        spread_number = n.split('_')[1].split('.png')[0]
        iso_number = m.split('_')[1].split('.png')[0]
        assert(int(spread_number.split('-')[1]) == int(iso_number.split('-')[1]) - 1)
        file = np.load(f'{self.root}/backgrounds/background_{fire_number}.npz')
        topology = np.concatenate([np.expand_dims(file["a1"], axis=0), np.expand_dims(file["a2"], axis=0), np.expand_dims(file["a3"], axis=0)
                                , np.expand_dims(file["a4"], axis=0), np.expand_dims(file["a5"], axis=0), 
                                np.expand_dims(file["a6"], axis=0), np.expand_dims(file["a7"], axis=0), 
                                np.expand_dims(file["a8"], axis=0)])
        spread = read_image(f"{self.root}/spreads/fire_{spread_number}.png")
        spread = torch.where(spread[1] == 231, 1.0, 0.0)
        isoc = read_image(f"{self.root}/spreads/iso_{iso_number}.png")
        isoc = torch.where(isoc[1] == 231, 1.0, 0.0).unsqueeze(0)
        input = torch.cat((spread.unsqueeze(0), torch.from_numpy(topology)))
        w_history = pd.read_csv(f'{self.root}/landscape/WeatherHistory.csv', header=None)
        n_weather = w_history.iloc[int(fire_number)-1].values[0].split("Weathers/")[1]
        weather = pd.read_csv(f'{self.root}/landscape/Weathers/' + n_weather)
        scenario_n = int(spread_number.split('-')[1]) 
        wind_speed = weather.iloc[int(scenario_n)]["WS"]
        wind_direction = weather.iloc[int(scenario_n)]["WD"]
        weather_tensor = torch.Tensor([wind_speed, wind_direction])
        if self.transform:
            input = self.transform(input)
            isoc = self.transform(isoc)
            weather_tensor = self.transform(weather_tensor)
        return (input, weather_tensor), isoc
            

        

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
        w_dirs = os.listdir(f"{self.root}/landscape/Weathers")
        self.max_wd = 0
        self.max_ws = 0
        self.min_wd = 100
        self.min_ws = 100
        for i in w_dirs:
            weather = pd.read_csv(f'{self.root}/landscape/Weathers/' + i)
            for row in weather.iterrows():
                wd = row[1]["WD"]
                ws = row[1]["WS"]
                if wd > self.max_wd:
                    self.max_wd = wd
                if ws > self.max_ws:
                    self.max_ws = ws
                if wd < self.min_wd:
                    self.min_wd = wd
                if ws < self.min_ws:
                    self.min_ws = ws
            

    def __call__(self, image):
        shape = image.shape
        if len(shape) > 2:
            if len(shape) > 3:
                for i in range(shape[0]):
                    image[i] = self(image[i])
            else:
                for i in range(1, shape[0]):
                    image[i].apply_(lambda x: (x + 1)/(self.maxes[i-1] + 1))
            return image
        else:
            if len(shape) > 1:
                for i in range(shape[0]):
                    image[i] = self(image[i])
            else:
                image[0] = (image[0] - self.min_ws) / (self.max_ws - self.min_ws)
                image[1] = (image[1] - self.min_wd) / (self.max_wd - self.min_wd)
            return image