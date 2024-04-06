import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


data = pd.read_csv('Data.csv')

RibeyeMeasurements = data["Ribeye"]
fatThickness = data.FatThickness

# Load the file
loaded_data = torch.load('converted_images.pt')

# Check the type and structure of the loaded data
print("Data type:", type(loaded_data))

if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
    x, y = loaded_data  # Unpack the tuple into x and y
    print("x type:", type(x))
    print("y type:", type(y))
    
    # Further inspect the contents of x and y
    if all(isinstance(item, torch.Tensor) for item in x) and all(isinstance(item, torch.Tensor) for item in y):
        # If x and y contain tensors, print their shapes
        for idx, tensor in enumerate(x):
            print(f"x[{idx}] shape:", tensor.shape)
        for idx, tensor in enumerate(y):
            print(f"y[{idx}] shape:", tensor.shape)
        
        # Define CustomDataset class
        class CustomDataset(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        # Create custom datasets for training and testing
        train_dataset = CustomDataset(x, y)
        
    else:
        print("x and y contain non-tensor items")
else:
    print("Invalid data structure")

# Define your neural network class and other training/test logic here
