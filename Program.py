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

import matplotlib.pyplot as plt

Data = pd.read_csv("Data.csv")

filenames = Data["Filename"].tolist()  # Convert column to a list
labels = Data["Grade Category"].tolist()  # Convert column to a list
one_hot_encoded = pd.get_dummies(Data['Grade Category'], prefix='Category')
#labels_new = F.one_hot(torch.tensor(labels), num_classes=10)

tensor = torch.tensor(one_hot_encoded.values)
labels = tensor.long()



# Determine the size of your images
sample_image_path = 'TestImages/00001312-1.tif'
sample_image = Image.open(sample_image_path)
image_width, image_height = sample_image.size

# data = pandas.read_csv('Data.csv')

# RibeyeMeasurements = data["Ribeye"]
# fatThickness = data.FatThickness

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



class MeatClassifier(nn.Module):
    def __init__(self):
        super(MeatClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * image_width//4 * image_height//4, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 output neurons for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * image_width//4 * image_height//4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Define hyperparameters


# Create DataLoader
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# Create the model
model = MeatClassifier()

# Define loss function and optimizer
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(image_width * image_height, 100)  # Assuming input images are of size image_width x image_height
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, image_width * image_height)  # Flatten the input image
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

f = MyNeuralNet()

def train_model(dl, f, labels_tensor, n_epochs=1):
    # Optimization
    opt = SGD(f.Parameters(), lr=0.00001)
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
            x = x.view(-1, image_width * image_height)  # Flatten the input image
            loss_value = L(f(x), labels_tensor) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch + i / N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)



# Call train_model function without slicing the dataset

epoch_data, loss_data = train_model(train_dl, labels, f)

# # Plotting code remains the same
# plt.plot(epoch_data, loss_data)
# plt.xlabel('Epoch Number')
# plt.ylabel('Cross Entropy')
# plt.title('Cross Entropy (per batch)')
# plt.show()


