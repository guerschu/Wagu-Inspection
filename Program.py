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



tif_dir = 'TrainImages'

# Directory to save .pt file
pt_file = 'converted_images.pt'

# Define transformation to apply to images (if needed)
transform = transforms.Compose([
    transforms.Resize(( 28, 28)),  # Resize image
    transforms.Grayscale(),
    transforms.ToTensor(),           # Convert to PyTorch tensor
    # Add more transformations as needed
])

# Load .tif files, apply transformations, and store as PyTorch tensors
images = []
for file in os.listdir(tif_dir):
    if file.endswith('.tif'):
        img = Image.open(os.path.join(tif_dir, file))
        img = transform(img)
        images.append(img)

# Concatenate the list of tensors into one tensor
images_tensor = torch.stack(images)

# Save the PyTorch tensor as .pt file
torch.save(images_tensor, pt_file)

x = torch.load('converted_images.pt')
y = torch.load('converted_images.pt')


y.shape

x.view(-1, 28**2).shape

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


train_ds = CTDataset('TrainImages')
test_ds = CTDataset('TestImages')

