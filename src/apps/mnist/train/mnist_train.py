import torch
import time
import os
import json
import sys
import random
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import argparse
import copy

from qlinear import QLinear

parser = argparse.ArgumentParser()
parser.add_argument("--W_bits", type=int, default=8)
parser.add_argument("--A_bits", type=int, default=8)
parser.add_argument("--logfile", type=str, default="savedata/logfiles/default_logfile")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--model_dir", type=str, default="savedata/models/default_model_dir")
parser.add_argument("--lr", type=float, default=1e-4)

args = parser.parse_args()

epochs = args.epochs
W_bits = args.W_bits
A_bits = args.A_bits
outfile = args.logfile
model_dir = args.model_dir

if not os.path.exists(os.path.dirname(outfile)):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

class QuantizedNN(torch.nn.Module):
    def __init__(self, hidden_size, W_bits=8, A_bits=4):
        super().__init__()
        self.L1 = QLinear(28*28, hidden_size, W_bits=W_bits, A_bits=A_bits)
        self.L2 = QLinear(hidden_size, hidden_size, W_bits=W_bits, A_bits=A_bits)
        self.L3 = QLinear(hidden_size, 10, W_bits=W_bits, A_bits=A_bits)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        return x

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 784
hidden_size = 4096
num_classes = 10
num_epochs = args.epochs
batch_size = 100
lr = args.lr

# Create model
#model = QuantizedNN(hidden_size, W_bits=W_bits, A_bits=A_bits)
model = nn.Sequential(
    QLinear(28*28, hidden_size, W_bits=W_bits, A_bits=A_bits, method="fake"), nn.ReLU(inplace=True),
    QLinear(hidden_size, hidden_size, W_bits=W_bits, A_bits=A_bits, method="fake"), nn.ReLU(inplace=True),
    QLinear(hidden_size, 10, W_bits=W_bits, A_bits=A_bits, method="fake"),
).to(device)

model.to(device)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def mnist_eval():
    with torch.no_grad():
        correct = 0
        total = 1
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100 * correct / total


# Train the model
total_step = len(train_loader)
data = {
    "W_bits" : W_bits,
    "A_bits" : A_bits,
    "accs" : [],
    "hidden_size" : hidden_size,
    "epochs" : epochs,
    "lr" : lr
}

best_model = None
t_prev = time.time()
for epoch in range(num_epochs):
    acc = mnist_eval()
    print(acc)
    data["accs"].append(acc)
    
    if acc == max(data["accs"]):
        best_model = copy.deepcopy(model)

    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            t_elapsed = time.time()-t_prev
            t_prev = time.time()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} (t_elapse: {:.4f})'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(),  t_elapsed))
            sys.stdout.flush()


with open(outfile, "w") as f:
    f.write(json.dumps(data))

name = "W_bits=%d_A_bits=%d.ckpt" % (W_bits, A_bits)
torch.save(best_model.state_dict(), '%s/%s' % (model_dir, name))
