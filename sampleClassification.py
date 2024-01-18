import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Iris dataset
# Assuming 'iris.csv' is the path to the downloaded Iris dataset CSV file
csv_file = './datasets/Iris.csv'

# Load the Iris dataset from a CSV file
df = pd.read_csv(csv_file)
print(df.head())


# Encode the species column to integers
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Separate features and target
features = df.drop(['Id', 'Species'], axis=1).values
target = df['Species'].values

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.long)

# Create a custom dataset
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
train_dataset = IrisDataset(x_train, y_train)
test_dataset = IrisDataset(x_test, y_test)

#Create DataLoaders for train and test sets
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# Neural network model
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 4 input features, 10 neurons in the hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # 3 output classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the network
model = IrisNet()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Print training progress
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate and print the accuracy
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Calculate and print precision, recall, and F1 score
precision = precision_score(all_targets, all_predictions, average='macro')
recall = recall_score(all_targets, all_predictions, average='macro')
f1 = f1_score(all_targets, all_predictions, average='macro')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save the model weights
torch.save(model.state_dict(), './models/iris.pth')  # .pth is the recommended extension
