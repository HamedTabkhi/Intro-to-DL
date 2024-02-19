import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Sample text
text = "This is a simple example to demonstrate how to predict the next character using RNN in PyTorch."

# Creating character vocabulary
# part of the data preprocessing step for a character-level text modeling task. 
# Create mappings between characters in the text and numerical indices

#set(text): Creates a set of unique characters found in the text. The set function removes any duplicate characters.
#list(set(text)): Converts the set back into a list so that it can be sorted. 
# sorted(list(set(text))): Sorts the list of unique characters. 
chars = sorted(list(set(text)))
#This line creates a dictionary that maps each character to a unique index (integer)."
ix_to_char = {i: ch for i, ch in enumerate(chars)}
#Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
char_to_ix = {ch: i for i, ch in enumerate(chars)} 
chars = sorted(list(set(text)))




# Preparing the dataset
max_length = 10  # Maximum length of input sequences
X = []
y = []
for i in range(len(text) - max_length):
    sequence = text[i:i + max_length]
    label = text[i + max_length]
    X.append([char_to_ix[char] for char in sequence])
    y.append(char_to_ix[label])

X = np.array(X)
y = np.array(y)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Defining the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        #This line takes the input tensor x, which contains indices of characters, and passes it through an embedding layer (self.embedding). 
        #The embedding layer converts these indices into dense vectors of fixed size. 
        #These vectors are learned during training and can capture semantic similarities between characters. 
        #The result is a higher-dimensional representation of the input sequence, where each character index is replaced by its corresponding embedding vector. 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        #The RNN layer returns two outputs: 
        #1- the output tensor containing the output of the RNN at each time step for each sequence in the batch, 
        #2-the hidden state (_) of the last time step (which is not used in this line, hence the underscore).
        output, _ = self.rnn(embedded)
        #The RNN's output contains the outputs for every time step, 
        #but for this task, we're only interested in the output of the last time step because we're predicting the next character after the sequence. 
        #output[:, -1, :] selects the last time step's output for every sequence in the batch (-1 indexes the last item in Python).
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 100

# Model, loss, and optimizer
model = CharRNN(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        #The use of the underscore _ is a common Python convention to indicate that the actual maximum values returned by torch.max are not needed and can be disregarded. 
        #What we are interested in is the indices of these maximum values, which are captured by the variable predicted. These indices represent the model's predictions for each example in the validation set.
        _, predicted = torch.max(val_output, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "This is a simple example to demonstrate how to predict the next char"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")