import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets

# 1. Data PreProcessing
X_np , y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Convert to Torch.tensor, astype:because we want to convert to float32
X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
# Reshaping y tensor, as it has only 1 raw and we are converting it to column vector
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 2. Model
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)


# 2. Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 3. Training
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)
    # Loss
    loss = criterion(y_pred, y)
    
    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    # zero gradient
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

# Plot
predicted = model(X).detach() # set gradient attribute to False
plt.plot(X_np, y_np, 'r+')
plt.plot(X_np, predicted, 'b')
plt.show()