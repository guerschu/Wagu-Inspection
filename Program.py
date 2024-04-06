import pandas as pd
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import os
from torch.optim import SGD
from PIL import Image

import matplotlib.pyplot as plt

Data = pd.read_csv("annotations.csv")

filenames = Data["Filename"].tolist()  # Convert column to a list
print(type(filenames[0]))
labels = Data["Grade Category"].tolist()  # Convert column to a list
print(type(labels[0]))
one_hot_encoded = pd.get_dummies(Data['Grade Category'], prefix='Category')
#labels_new = F.one_hot(torch.tensor(labels), num_classes=10)

tensor = torch.tensor(one_hot_encoded.values)
labels = tensor.long()
print(labels.shape)



# Determine the size of your images
sample_image_path = 'TrainFiles/00000003-1.tif'
sample_image = Image.open(sample_image_path)
image_width, image_height = sample_image.size


class CTDataset(Dataset):
    def __init__(self, root_dir, filenames, labels, transform=None):
        self.root_dir = root_dir
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]  # Get the corresponding label from the labels list
        return image, label

    

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

batch_size = 10
learning_rate = 0.00001
num_epochs = 4

train_ds = CTDataset(root_dir="TrainFiles", filenames=filenames, transform=transform, labels=labels)
test_ds = CTDataset(root_dir="TestImages", filenames=filenames, labels=labels, transform=transform)

train_dl = DataLoader(train_ds, batch_size=5)

L = nn.CrossEntropyLoss()

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(image_width* image_height,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,image_width* image_height)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()
    



f = MyNeuralNet()

def train_model(dl, f, n_epochs=20):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad() 
            print(y.shape)
            print(f(x).shape)
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)

 


epoch_data, loss_data = train_model(train_dl, f)

plt.plot(epoch_data, loss_data)
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (per batch)')
