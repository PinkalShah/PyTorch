import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# GPU Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Paramertes
input_size = 784 # 28*28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001


# 1. Import Data
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor())


# 2. Data Loader
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Sample data
sample_data = iter(train_dataloader)
samples, labels = sample_data.next()
print(samples.shape, labels.shape)
# Sample.shape = (100, 1, 28, 28) , 100-sample according to batch size, 1- channel(No color), 28,28- image array 

# Plotting Sample data
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()


# 3. Multilayer Neural Network, Activation Function

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
         super(NeuralNetwork, self).__init__()
         
         self.l1 = nn.Linear(input_size, hidden_size)
         self.relu = nn.ReLU()
         
         self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)


# 4.Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 5. Traning Loop with batch
num_total_steps = len(train_dataloader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # Re-shaping images, current 100,1,28,28 
        # input_size = 784
        # requirement = 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss details
        if (i+1)% 100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{num_total_steps}, loss: {loss.item():.5f}')


# 6. Model Evaluation 
with torch.no_grad():
    num_current_prediction = 0
    num_samples = 0
    for images, labels  in test_dataloader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Value, index
        _ , predictions = torch.max(outputs, 1)
        num_samples += labels.shape[0]
        num_current_prediction += (predictions == labels).sum().item()

    accuracy = 100.0 * num_current_prediction/ num_samples
    print(f'Accuracy is {accuracy}')