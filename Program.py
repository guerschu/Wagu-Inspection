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

class CTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = self.load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

    def load_images(self):
        images = []
        for file in os.listdir(self.root_dir):
            if file.endswith('.tif'):
                images.append(file)
        return images

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_ds = CTDataset(root_dir='TrainImages', transform=transform)
test_ds = CTDataset(root_dir='TestImages', transform=transform)
