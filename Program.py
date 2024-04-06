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

# x = torch.load('TrainImages/00001273-2.tif')
# y = torch.load('TrainImages/00001273-2.tif')


class CTDataset(Dataset):
    # def __init__(self, filepath):
    #     self.x, self.y = torch.load(filepath)
    #     for i in range(len(self.x)):  # Iterate over the indices of the tensor x
    #         for j in range(len(self.x[i-1])):
    #             print(self.x[i])
    #         self.x[i] += self.x[i] / 255.  # Divide each element by 255
    #     self.y = F.one_hot(self.y, num_classes=10).to(torch.float32)  # Convert y to one-hot and float32

        # Directory containing the TIFF images
    def __init__(self, filepath):
             
        tif_dir = filepath

        # List to store loaded images
        images = []

        # Iterate over TIFF files in the directory
        for file in os.listdir(tif_dir):
            if file.endswith('.tif'):
                # Load the TIFF image using PIL
                img = Image.open(os.path.join(tif_dir, file))
                # Convert the image to a PyTorch tensor
                img_tensor = transforms.ToTensor()(img)
                # Append the tensor to the list of images
                images.append(img_tensor)

        # Stack the list of tensors into a single tensor
        images_tensor = torch.stack(images)
        print("Tensor shape:", images_tensor.shape)  # Print tensor shape (batch_size, channels, height, width)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]



train_ds = CTDataset('TrainImages')

test_ds = CTDataset('TrainImages')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),           # Convert to PyTorch tensor
    # Add more transformations as needed
])