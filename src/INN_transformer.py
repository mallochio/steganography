#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   INN_transformer.py 
@Time    :   2023/07/19 11:25:25
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Invertible vision transformer made with pytorch for steganography to hide a smaller image inside a larger image
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm


class VisionTransformer(nn.Module):
    def __init__(self, dim):
        super(VisionTransformer, self).__init__()
        self.inn = create_inn(dim)

    def forward(self, x):
        return self.inn(x)

    def inverse(self, y):
        return self.inn(y, rev=True)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def create_inn(dim):
    nodes = [Ff.InputNode(dim, name="input")]
    
    # specify the structure of your INN here
    # nodes.append(Fm.AllInOneBlock(dim))

    # nodes.append(Fm.GLOWCouplingBlock(in_channels=dim, Fm.AllInOneBlock(dim)))

    
    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)



def hide_image(larger_image, smaller_image):
    # implement here

def reveal_image(larger_image):
    # implement here


def train(model, dataloader, optimizer, device):
    model.train()
    for images, _ in dataloader:
        images = images.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
