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
import json
import time

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
        HOME_FOLDER = pathlib.Path(root + "/spreads_400/")
        self.fires = {}
        self.isoc = {}
        self.n = 0
        dir_list = set(list(i.name.split('_')[0].split('-')[0] for i in HOME_FOLDER.iterdir()))
        for item in HOME_FOLDER.iterdir():
            if "fire" in item.name:
                number = int(item.name.split("_")[1].split('-')[0])
                spread_number = int(item.name.split("_")[1].split('-')[1].split('.')[0])
                if number not in self.fires.keys():
                    self.fires[number] = [spread_number]
                else:
                    self.fires[number].append(spread_number)
            else:
                number = int(item.name.split("_")[1].split('-')[0])
                spread_number = int(item.name.split("_")[1].split('-')[1].split('.')[0])
                if number not in self.isoc.keys():
                    self.isoc[number] = [spread_number]
                else:
                    self.isoc[number].append(spread_number)
        self.n = 0
        self.keys = {}
        for i in self.fires:
            self.fires[i].sort()
            self.isoc[i].sort()
            for j in range(len(self.fires[i])):
                if j == len(self.fires[i]) - 1:
                    break
                else:
                    self.keys[self.n] = (i, j)
                    self.n += 1
        
        self.transform = tform
        self.data = {}
        with rioxarray.open_rasterio(f"{root}/landscape/Input_Geotiff.tif") as src:
            self.maxes = src.max(dim=["x", "y"]).values
        with rasterio.open(root + '/landscape/Input_Geotiff.tif') as f:
            self.band_0 = f.read(1)
            self.band_0 = np.where(self.band_0 == -9999.0, -1, self.band_0) / self.maxes[0]
            self.band_1 = f.read(2)
            self.band_1 = np.where(self.band_1 == -9999.0, -1, self.band_1) / self.maxes[1]
            self.band_2 = f.read(3)
            self.band_2 = np.where(self.band_2 == -9999.0, -1, self.band_2) / self.maxes[2]
            self.band_3 = f.read(4)
            self.band_3 = np.where(self.band_3 == -9999.0, -1, self.band_3) / self.maxes[3]
            self.band_4 = f.read(5)
            self.band_4 = np.where(self.band_4 == -9999.0, -1, self.band_4) / self.maxes[4]
            self.band_5 = f.read(6)
            self.band_5 = np.where(self.band_5 == -9999.0, -1, self.band_5) / self.maxes[5]
            self.band_6 = f.read(7)
            self.band_6 = np.where(self.band_6 == -9999.0, -1, self.band_6) / self.maxes[6]
            self.band_7 = f.read(8)
            self.band_7 = np.where(self.band_7 == -9999.0, -1, self.band_7) / self.maxes[7]

        
        self.landscape = np.stack([self.band_0, self.band_1, self.band_2, self.band_3, self.band_4, self.band_5, self.band_6, self.band_7], axis=0)
        with open(root + "/indices.json") as f:
            self.indices = json.load(f)
        
        self.w_history = pd.read_csv(f'{self.root}/landscape/WeatherHistory.csv', header=None)
        self.weathers = {}
        WEATHER_FOLDER = pathlib.Path(root + "/landscape/Weathers")
        for item in WEATHER_FOLDER.iterdir():
            self.weathers[item.name.split("/Weathers/")[0]] = pd.read_csv(f'{self.root}/landscape/Weathers/' + item.name)
        
        
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
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        fire_number, spread_number = self.keys[i]
        iso_number = spread_number + 1
        assert(spread_number == iso_number - 1)
        y, y_, x, x_ = self.indices[str(fire_number)]
        topology = self.landscape[:,y:y_, x:x_]
        spread = read_image(f"{self.root}/spreads_400/fire_{fire_number}-{spread_number}.png")
        spread = torch.where(spread[1] == 231, 1.0, 0.0)
        isoc = read_image(f"{self.root}/spreads_400/iso_{fire_number}-{iso_number}.png")
        isoc = torch.where(isoc[1] == 231, 1.0, 0.0).unsqueeze(0)
        input = torch.cat((spread.unsqueeze(0), torch.from_numpy(topology)))
        n_weather = self.w_history.iloc[int(fire_number)-1].values[0].split("Weathers/")[1]
        weather = self.weathers[n_weather]
        scenario_n = spread_number 
        wind_speed = (weather.iloc[scenario_n]["WS"] - self.min_ws) / (self.max_ws - self.min_ws)
        wind_direction = (weather.iloc[scenario_n]["WD"] - self.min_wd) / (self.max_wd - self.min_wd)
        weather_tensor = torch.Tensor([wind_speed, wind_direction])
        if self.transform:
            input = self.transform(input)
            isoc = self.transform(isoc)
            weather_tensor = self.transform(weather_tensor)
        return (fire_number, iso_number), (input, weather_tensor), isoc
            

        