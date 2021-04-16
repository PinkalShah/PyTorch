import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms

# GPU Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Paramertes
num_epochs = 100
batch_size = 10
learning_rate = 0.001


# 1. Import Data
# Dataset has image of range [0, 1] so we are normalizing them within the range of [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


# 2. Data Loader
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

''' 
Image shape = 3*32*32
After convo1 = 6*28*28 based on 
((W-F+2P)/S + 1) where W = input, F = filter, P = padding, S = stride
After pool1 = 6*14*14 because stride is 2
After convo2 = 16*10*10
After pool2 = 16*5*5 because stride is 2
flattining = 400
fc1 = 120
fc2 = 84
fc3 = 10
'''

# 3. Convonutional Neural Network
class ConvoNN(nn.Module):
    def __init__(self):
        super(ConvoNN, self).__init__()
        self.convo1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convo2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120) #
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.convo1(x)))
        x = self.pool(F.relu(self.convo2(x)))
        # flattning
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvoNN().to(device)

# 4.Loss and Optimizer
criterion = nn.CrossEntropyLoss() # this includes sigmoid
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# 5. Traning Loop with batch
num_total_steps = len(train_dataloader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # input_size = [4, 3, 32, 32] = 4, 3, 1024
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
        # loss details
        if (i+1)% 500 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{num_total_steps}, loss: {loss.item():.5f}')


# 6. Model Evaluation 
with torch.no_grad():
    num_current_prediction = 0
    num_samples = 0
    num_class_correct = [0 for i in range(10)]
    num_class_samples = [0 for i in range(10)]

    for images, labels  in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Value, index
        _ , prediction = torch.max(outputs, 1)
        num_samples += labels.shape[0]
        num_current_prediction += (prediction == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = prediction[i]
            if(label == pred):
                num_class_correct[label] += 1
            num_class_samples[label] += 1

    accuracy = 100.0 * num_current_prediction/ num_samples
    print(f'Accuracy is {accuracy}')

 