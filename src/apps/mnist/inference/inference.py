import torch
import math
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
parser.add_argument("--W_bits", type=int, default=32)
parser.add_argument("--A_bits", type=int, default=32)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--method", type=str, default="cutlass", choices=["cutlass","pbatch","fake"])

args = parser.parse_args()

W_bits = args.W_bits
A_bits = args.A_bits
model_path = args.model_path
method = args.method

assert(model_path is not None and os.path.exists(model_path))

class QuantizedNN(torch.nn.Module):
    def __init__(self, hidden_size, W_bits=8, A_bits=4):
        super().__init__()
        self.L1 = QLinear(28*28, hidden_size, W_bits=W_bits, A_bits=A_bits, method=method)
        self.L2 = QLinear(hidden_size, hidden_size, W_bits=W_bits, A_bits=A_bits, method=method)
        self.L3 = QLinear(hidden_size, 10, W_bits=W_bits, A_bits=A_bits, method=method)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        return x

# Check Device configuration
device = torch.device('cuda')

# Define Hyper-parameters
input_size = 784
hidden_size = 4096
num_classes = 10
batch_size = 1

# Create model
#model = QuantizedNN(hidden_size, W_bits=W_bits, A_bits=A_bits)
model = nn.Sequential(
    QLinear(28*28, hidden_size, W_bits=W_bits, A_bits=A_bits, method=args.method, optimize_cutoff=False), nn.ReLU(inplace=True),
    QLinear(hidden_size, hidden_size, W_bits=W_bits, A_bits=A_bits, method=args.method, optimize_cutoff=False), nn.ReLU(inplace=True),
    QLinear(hidden_size, 10, W_bits=W_bits, A_bits=A_bits, method=args.method, optimize_cutoff=False),
).cuda()
model.to(device)

# Load model
state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)
model.eval()

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

def mnist_eval():
    with torch.no_grad():
        correct = 0
        total = 1
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)

            # For cutlass, pad image to 256
            if args.method == "cutlass":
                n_to_pad = math.ceil(28*28/256)*256 - 28*28
                images = F.pad(images, pad=(0,n_to_pad), mode="constant", value=0)
            
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100 * correct / total

def performance_eval():
    img = next(iter(test_loader))[0].reshape(-1, 28*28).to(device)
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        tstart = time.time()
        for j in range(100):
            model(img)
        torch.cuda.synchronize()
        tend = time.time()
        times.append(tend-tstart)
    ttimes = min(times)
    return ttimes

acc = mnist_eval()
elapsed = performance_eval()

print("module,mnist,%s,%d,%d,%f,%f" % (method, W_bits, A_bits, acc,elapsed))


