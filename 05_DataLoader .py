''' 
epoch = one forward and backwad pass for all training samples
batch_size = number of training sample in one forward and backward pass
number_of_iterations = number of passes, each pass using batch_size (number of samples)
For example: 100 samples, batch_size=20 then we take 5 iteratios for 1 epoch'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math

class WineDataset(Dataset):
    def __init__(self):
        data = np.loadtxt(r'L_PyTorch\wine.csv',delimiter=',',dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(data[:,1:])
        self.y = torch.from_numpy(data[:,[0]])
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
dataiterator = iter(dataloader)
data = dataiterator.next()
features, labels = data
print(features,'\n', labels)

# Training Loop - Temperory
num_epochs = 10
total_samples = len(dataset)
num_iterations = math.ceil(total_samples/5) # 2 i  batch_size
print(total_samples, num_iterations)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        if (i+1) % 5 ==0:
            print(f'epoch: {epoch+1}/ {num_epochs}, step {i+1}/{num_iterations}, input: {inputs.shape}')
        # backward pass
        # update

