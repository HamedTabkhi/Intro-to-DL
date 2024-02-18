import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot


# Check for CUDA support and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a sine wave dataset
timesteps = 1000   # Total timesteps
time = np.linspace(0, 40, timesteps)  # Time variable
data = np.sin(time)  # Sine wave dataset

# Convert data to PyTorch tensors and move to the selected device (GPU or CPU)
data = torch.FloatTensor(data).view(-1, 1).to(device)

# Define a simple RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output

    def forward(self, x):
        out, _ = self.rnn(x)  # Process input through RNN
        out = self.fc(out[:, -1, :])  # Pass the last time step output to linear layer
        return out

# Model parameters
input_size = 1
hidden_size = 100
output_size = 1

# Instantiate the model and move it to the selected device
model = RNNModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()  # Loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Optimizer

# Prepare data for training by creating input-output sequences
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

seq_length = 50  # Length of the sequence of data points considered for prediction
train_data = create_inout_sequences(data, seq_length)

# Training loop
epochs = 40
for epoch in range(epochs):
    for seq, labels in train_data:
        optimizer.zero_grad()
        seq = seq.view(1, seq_length, -1).to(device)  # Ensure data is on the same device as model
        labels = labels.to(device)  # Move labels to device
        y_pred = model(seq)  # Forward pass
        loss = criterion(y_pred, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Predict future values
with torch.no_grad():
    test_seq = data[:seq_length].view(1, seq_length, -1).to(device)  # Prepare initial sequence for prediction
    preds = []
    for _ in range(200):
        y_test_pred = model(test_seq)
        preds.append(y_test_pred.item())
        new_seq = test_seq[:, 1:, :]
        test_seq = torch.cat((new_seq, y_test_pred.view(1, 1, 1)), 1)

# Move predictions and original data back to CPU for plotting
preds = torch.tensor(preds).view(-1, 1).cpu()
data = data.cpu()

# Plot the results and save the plot
plt.figure(figsize=(10,5))
plt.plot(time, data.numpy().flatten(), label='Ground Truth')  # Plot the full ground truth sine wave
plt.plot(time[seq_length:seq_length+200], preds.numpy().flatten(), label='Predicted')  # Plot the predicted values
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sine Wave Prediction')
plt.savefig('./temp/sine_wave_prediction_gpu.png')  # Save the plot as a file
