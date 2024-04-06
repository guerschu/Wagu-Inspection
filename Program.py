
import numpy as np
import pandas 
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

data = pandas.read_csv('Data.csv')

RibeyeMeasurements = data.Ribeye

fatThickness = data.FatThickness

