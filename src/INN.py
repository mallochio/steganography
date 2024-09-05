#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   mixed_edge_detection.py
@Time    :   2023/07/19 10:07:39
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   A steganography method for hiding a smaller image inside a larger image using invertible neural networks (INNs) made with pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



class InvBlock(nn.Module):
    # Invertible Block (This can be stacked to form the whole architecture)
    def __init__(self):
        super(InvBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 1)
        self.conv2 = nn.Conv2d(128, 128, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x1 = x[:, :C//2, :, :]
        x2 = x[:, C//2:, :, :]

        Fx1 = self.conv1(x1)
        y1 = x1 + Fx1
        Fy1 = self.conv2(y1)
        y2 = x2 + Fy1
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        B, C, H, W = y.size()
        y1 = y[:, :C//2, :, :]
        y2 = y[:, C//2:, :, :]

        Fy1 = self.conv2(y1)
        x2 = y2 - Fy1
        Fx1 = self.conv1(x2)
        x1 = y1 - Fx1
        return torch.cat([x1, x2], dim=1)



class InvNet(nn.Module):
    # Whole Invertible Neural Network (Add more blocks as necessary)
    def __init__(self):
        super(InvNet, self).__init__()
        self.block1 = InvBlock()
        # you can add more blocks as necessary
        
    def forward(self, x):
        return self.block1(x)
        
    def inverse(self, y):
        return self.block1.inverse(y)
        

# Loss function
def loss_fn(x, y):
    return torch.mean(torch.abs(x-y))


# Training function
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        y = model(data)
        loss = loss_fn(y, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

# Testing function
def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            y = model(data)
            test_loss += loss_fn(y, data).item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))


# Main function
def main():
    # Setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setting up the hyperparameters
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3

    # Loading the dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initializing the model
    model = InvNet().to(device)

    # Setting up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)

    # Saving the model
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
