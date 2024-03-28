# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

# Dataset containing pairs of synonyms
dataset = [
    # Each tuple contains two words that are synonyms
    ("big", "large"), ("small", "tiny"), ("happy", "joyful"), ("sad", "unhappy"),
    ("fast", "quick"), ("slow", "sluggish"), ("bright", "luminous"), ("dark", "dim"),
    ("hard", "difficult"), ("easy", "simple"), ("hot", "warm"), ("cold", "chilly"),
    ("old", "ancient"), ("new", "recent"), ("good", "excellent"), ("bad", "poor"),
    ("strong", "powerful"), ("weak", "feeble"), ("tall", "high"), ("short", "brief"),
    ("fat", "obese"), ("thin", "slender"), ("rich", "wealthy"), ("poor", "needy"),
    ("smart", "intelligent"), ("dumb", "stupid"), ("happy", "content"), ("sad", "mournful"),
    ("beautiful", "attractive"), ("ugly", "unattractive"), ("expensive", "costly"),
    ("cheap", "inexpensive"), ("empty", "vacant"), ("full", "filled"), ("noisy", "loud"),
    ("quiet", "silent")
]

# Special tokens for the start and end of sequences
SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token
max_length = 12


# Preparing the character to index mapping and vice versa
# These mappings will help convert characters to numerical format for the neural network
# 'SOS' and 'EOS' tokens are added at the start of the char_to_index dictionary
char_to_index = {"SOS": SOS_token, "EOS": EOS_token, **{char: i+2 for i, char in enumerate(sorted(list(set(''.join([word for pair in dataset for word in pair])))))}}
index_to_char = {i: char for char, i in char_to_index.items()}

class SynonymDataset(Dataset):
    """Custom Dataset class for handling synonym pairs."""
    def __init__(self, dataset, char_to_index):
        self.dataset = dataset
        self.char_to_index = char_to_index

    def __len__(self):
        # Returns the total number of synonym pairs in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieves a synonym pair by index, converts characters to indices,
        # and adds the EOS token at the end of each word.
        input_word, target_word = self.dataset[idx]
        input_tensor = torch.tensor([self.char_to_index[char] for char in input_word] + [EOS_token], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_index[char] for char in target_word] + [EOS_token], dtype=torch.long)
        return input_tensor, target_tensor

# Creating a DataLoader to batch and shuffle the dataset
synonym_dataset = SynonymDataset(dataset, char_to_index)
dataloader = DataLoader(synonym_dataset, batch_size=1, shuffle=True)

# Setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#The encoder remains largely unchanged but we ensure it outputs all of its hidden states, 
#as these are needed for the attention mechanism in the decoder.
    
class Encoder(nn.Module):
    """Encoder that processes the input sequence and returns its hidden states."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # Embed input characters
        embedded = self.embedding(input).view(1, 1, -1)
        # LSTM over the embedded input
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        # Initial hidden state
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))


#The decoder uses the attention mechanism to focus on different parts of the input sequence for each step of the output sequence.
class AttnDecoder(nn.Module):
    """Decoder with attention mechanism."""
    def __init__(self, hidden_size, output_size, max_length=12, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Attention weights
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # Combine embedded input and context vector
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculate attention weights
        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        # Apply attention weights to encoder outputs to get context
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = torch.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))


# Assuming all characters in the dataset + 'SOS' and 'EOS' tokens are included in char_to_index
input_size = len(char_to_index)
hidden_size = 12
output_size = len(char_to_index)

encoder = Encoder(input_size=len(char_to_index), hidden_size=256).to(device)
decoder = AttnDecoder(hidden_size=256, output_size=len(char_to_index)).to(device)

# Set the learning rate for optimization
learning_rate = 0.01

# Initializing optimizers for both encoder and decoder with Stochastic Gradient Descent (SGD)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=12):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Encode each character in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[char_to_index['SOS']]], device=device)

    # Initial decoder hidden state is encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoder with attention
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == char_to_index['EOS']:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
n_epochs = 41

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        # Move tensors to the correct device
        input_tensor = input_tensor[0].to(device)
        target_tensor = target_tensor[0].to(device)
        
        # Perform a single training step and update total loss
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss
    
    # Print loss every 10 epochs
    if epoch % 10 == 0:
       print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')

def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, n_examples=5):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            loss = 0

            # Encode input
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decode with attention
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            total_loss += loss.item() / target_length
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            if i < n_examples:
                predicted_string = ''.join([index_to_char[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_string = ''.join([index_to_char[index.item()] for index in target_tensor if index.item() not in (SOS_token, EOS_token)])
                input_string = ''.join([index_to_char[index.item()] for index in input_tensor if index.item() not in (SOS_token, EOS_token)])
                
                print(f'Input: {input_string}, Target: {target_string}, Predicted: {predicted_string}')
        
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')

# Perform evaluation with examples
evaluate_and_show_examples(encoder, decoder, dataloader, criterion)
