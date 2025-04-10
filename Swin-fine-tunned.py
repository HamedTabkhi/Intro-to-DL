import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification, SwinConfig, AutoImageProcessor
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 5  # Fewer epochs for fine-tuning
batch_size = 32
learning_rate = 2e-5  # Smaller learning rate for fine-tuning
image_size = 32  # Swin expects 224x224 input by default

# Data preparation with proper preprocessing for Swin
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained Swin Transformer
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=10,  # CIFAR-10 has 10 classes
    ignore_mismatched_sizes=True  # Allows replacing the original classifier head
).to(device)

# Freeze backbone parameters and only train the classification head
for param in model.swin.parameters():
    param.requires_grad = False

# Only the classifier head will be trained
for param in model.classifier.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Training function
def train():
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                progress_bar.set_postfix({'loss': loss.item()})

# Testing function
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

# Run training and testing
if __name__ == '__main__':
    print("Downloading pretrained weights and starting fine-tuning...")
    train()
    print("\nEvaluating fine-tuned model...")
    test()