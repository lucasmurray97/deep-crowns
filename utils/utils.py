import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, root, tform=None, normalize = False):
    super(MyDataset, self).__init__()
    self.root = root
    self.tform = tform


  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    if not self.normalize:
      return np.expand_dims(self.data[i][0].astype('float32'), axis=0), np.array(self.rewards[i], dtype="float32")
    else:
      return np.expand_dims(self.data[i][0].astype('float32'), axis=0), np.array(self.scale(self.rewards[i]), dtype="float32")
  
  
  
def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images,r, model, n_images ,n_rows):

    with torch.no_grad():

        images = images
        images = model(images, r)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torch.round(torchvision.utils.make_grid(images[0:n_images], n_rows)).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

def visualise_output_reward(images,r, model, n_images ,n_rows):

    with torch.no_grad():

        images = images
        images, r = model(images, r)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torch.round(torchvision.utils.make_grid(images[0:n_images], n_rows)).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()
