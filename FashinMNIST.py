import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
import numpy as np

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Randomly select some samples from the training set
num_samples = 5
indices = random.sample(range(len(train_dataset)), num_samples)
samples, labels = zip(*[train_dataset[i] for i in indices])

# Class labels for FashionMNIST
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot and save the images with labels
fig, axs = plt.subplots(1, num_samples, figsize=(15, 15))
for i, (sample, label) in enumerate(zip(samples, labels)):
    axs[i].imshow(sample.squeeze(), cmap='gray')
    axs[i].set_title(f'Label: {class_labels[label]}')
    axs[i].axis('off')

# Save the figure
plt.savefig('./temp/fashion_mnist_samples.png')
plt.show()
plt.close()

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Neural network model
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)  # Image size is 28x28
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 classes in FashionMNIST

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the model and move it to the GPU if available
model = FashionMNISTNet().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
         # Move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Print training progress
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation loop
model.eval()  # Set the model to evaluation mode
all_predictions = []
all_targets = []
with torch.no_grad():  # Disable gradient calculation
    for inputs, targets in test_loader:
        # Move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
        all_predictions.extend(predicted.tolist())
        all_targets.extend(targets.tolist())

# Calculate accuracy, precision, recall, and F1 score
accuracy = 100 * sum([p == t for p, t in zip(all_predictions, all_targets)]) / len(all_targets)
precision = precision_score(all_targets, all_predictions, average='weighted')
recall = recall_score(all_targets, all_predictions, average='weighted')
f1 = f1_score(all_targets, all_predictions, average='weighted')

# Print the metrics
print(f'Accuracy: {accuracy}%')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')