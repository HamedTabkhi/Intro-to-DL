{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 408
        },
        "id": "bkC8EA9zJi4e",
        "outputId": "fe70873a-03df-41eb-e28e-cf126c63d11c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 0, Loss: 0.0024328995496034622\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-22195859b69d>\u001b[0m in \u001b[0;36m<cell line: 77>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     82\u001b[0m         \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseq\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# Generate predictions\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     83\u001b[0m         \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# Compute loss\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 84\u001b[0;31m         \u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# Backpropagation\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     85\u001b[0m         \u001b[0moptimizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# Update model parameters\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     86\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mepoch\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;36m10\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/torch/_tensor.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(self, gradient, retain_graph, create_graph, inputs)\u001b[0m\n\u001b[1;32m    520\u001b[0m                 \u001b[0minputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    521\u001b[0m             )\n\u001b[0;32m--> 522\u001b[0;31m         torch.autograd.backward(\n\u001b[0m\u001b[1;32m    523\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgradient\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    524\u001b[0m         )\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/torch/autograd/__init__.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)\u001b[0m\n\u001b[1;32m    264\u001b[0m     \u001b[0;31m# some Python versions print out the first line of a multi-line function\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    265\u001b[0m     \u001b[0;31m# calls in the traceback and some print out the last line\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 266\u001b[0;31m     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\n\u001b[0m\u001b[1;32m    267\u001b[0m         \u001b[0mtensors\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    268\u001b[0m         \u001b[0mgrad_tensors_\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Setting up the device for TensorFlow operations depending on availability of CUDA (GPU support)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "# Generate a sine wave dataset\n",
        "timesteps = 1000  # Define the total number of timesteps for the sine wave\n",
        "time = np.linspace(0, 40, timesteps)  # Create a time array that is linearly spaced between 0 and 40\n",
        "data = np.sin(time)  # Generate sine wave values based on the time array\n",
        "\n",
        "# Convert the numpy data array into a PyTorch tensor, reshape it to be two-dimensional, and transfer to the specified device\n",
        "data = torch.FloatTensor(data).view(-1, 1).to(device)\n",
        "\n",
        "# Positional Encoding module to add information about the order of the tokens in the transformer model\n",
        "class PositionalEncoding(nn.Module):\n",
        "    def __init__(self, d_model, max_len=5000):\n",
        "        super(PositionalEncoding, self).__init__()\n",
        "        self.encoding = torch.zeros(max_len, d_model)  # Create a zero matrix of size [max_len x d_model]\n",
        "        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Create a position array\n",
        "        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # Compute the division term\n",
        "        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices\n",
        "        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices\n",
        "        self.encoding = self.encoding.unsqueeze(0)  # Add a batch dimension\n",
        "\n",
        "    def forward(self, x):\n",
        "        return x + self.encoding[:, :x.size(1)].detach()  # Add positional encodings to the input embeddings\n",
        "\n",
        "# Transformer model definition\n",
        "class TransformerModel(nn.Module):\n",
        "    def __init__(self, input_size, hidden_size, output_size, n_layers, n_heads):\n",
        "        super(TransformerModel, self).__init__()\n",
        "        self.input_emb = nn.Linear(input_size, hidden_size)  # Linear layer to project input to hidden size\n",
        "        self.pos_encoder = PositionalEncoding(hidden_size)  # Positional encoding layer\n",
        "        self.encoder_layer = nn.TransformerEncoderLayer(\n",
        "            d_model=hidden_size, nhead=n_heads, batch_first=True)  # Transformer encoder layer configuration\n",
        "        self.transformer_encoder = nn.TransformerEncoder(\n",
        "            self.encoder_layer, num_layers=n_layers)  # Stacking multiple transformer encoder layers\n",
        "        self.fc = nn.Linear(hidden_size, output_size)  # Final linear layer to map from hidden size to output size\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.input_emb(x)  # Apply linear transformation\n",
        "        x = self.pos_encoder(x)  # Add positional encoding\n",
        "        x = self.transformer_encoder(x)  # Pass through the transformer encoder\n",
        "        x = self.fc(x)  # Apply final linear transformation to each token\n",
        "        return x\n",
        "\n",
        "# Set model parameters\n",
        "input_size = 1\n",
        "hidden_size = 128\n",
        "output_size = 1\n",
        "n_layers = 2\n",
        "n_heads = 2\n",
        "\n",
        "# Instantiate the model and move it to the device\n",
        "model = TransformerModel(input_size, hidden_size, output_size, n_layers, n_heads).to(device)\n",
        "criterion = nn.MSELoss()  # Mean Squared Error Loss\n",
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer\n",
        "\n",
        "# Function to create input-output sequences of specified length\n",
        "def create_inout_sequences(input_data, tw):\n",
        "    inout_seq = []\n",
        "    L = len(input_data)\n",
        "    for i in range(L-tw):\n",
        "        train_seq = input_data[i:i+tw]  # Input sequence slice\n",
        "        train_label = input_data[i+1:i+tw+1]  # Output sequence slice, offset by one time step\n",
        "        inout_seq.append((train_seq, train_label))\n",
        "    return inout_seq\n",
        "\n",
        "seq_length = 10  # Length of the sequence of data points considered for prediction (number of tokens)\n",
        "train_data = create_inout_sequences(data, seq_length)  # Prepare data for training\n",
        "\n",
        "# Training loop\n",
        "epochs = 100\n",
        "for epoch in range(epochs):\n",
        "    for seq, labels in train_data:\n",
        "        optimizer.zero_grad()  # Clear gradients\n",
        "        seq = seq.view(1, seq_length, -1).to(device)  # Reshape and transfer sequence data to device\n",
        "        labels = labels.view(1, seq_length, -1).to(device)  # Reshape and transfer label data to device\n",
        "        y_pred = model(seq)  # Generate predictions\n",
        "        loss = criterion(y_pred, labels)  # Compute loss\n",
        "        loss.backward()  # Backpropagation\n",
        "        optimizer.step()  # Update model parameters\n",
        "    if epoch % 10 == 0:\n",
        "        print(f'Epoch {epoch}, Loss: {loss.item()}')  # Print loss every 10 epochs\n"
      ]
    }
  ]
}