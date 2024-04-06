import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import os

data = pandas.read_csv('Data.csv')

RibeyeMeasurements = data["Ribeye"]
fatThickness = data.FatThickness

x = torch.load('converted_images.pt')
y = torch.load('converted_images.pt')

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        for i in range(len(self.x)):  # Iterate over the indices of the tensor x
            print(self.x)
            self.x[i] += self.x[i] / 255.  # Divide each element by 255
        self.y = F.one_hot(self.y, num_classes=10).to(torch.float32)  # Convert y to one-hot and float32
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]



train_ds = CTDataset('converted_images.pt')

test_ds = CTDataset('TestingImages.pt')
