import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets


# 1. Data PreProcessing
# Import Data
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

n_samples, n_features = X.shape
print("Total Samples", n_samples)
print("Total Features", n_features)

# Converting to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Converting to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test= torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# Reshaping y tensor, as it has only 1 raw and we are converting it to column vector
y_train = y_train.view(y_train.shape[0], 1)
y_test =  y_test.view(y_test.shape[0], 1)


# 2. Model
# f = wx + b, sigmoid activation at last layer
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        
        self.liner = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred =torch.sigmoid(self.liner(x))
        return y_pred

model = LogisticRegression(n_features) # So layer size is 30 * 1


# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training Loop
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass and Loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()
    
    # zero gradient
    optimizer.zero_grad()

    if (epoch +1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.5f}')

# Evalutation should not be part of our computation so 
# Accuracy
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = y_pred.round()
    accuracy = y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy is: {accuracy:.4f}')