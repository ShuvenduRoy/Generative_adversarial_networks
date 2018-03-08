import torch
import os, time, pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import *


# Generator network
class generator(nn.Module):
    def __init__(self, dataset='mnist'):
        super(generator, self).__init__()
        self.image_height = 28
        self.image_width = 28
        self.input_dim = 62  # features in latent dimension
        self.output_dim = 1  # number of output channels

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Linear(128 * (self.image_height // 4) * (self.image_width // 4)),
            nn.BatchNorm1d(128 * (self.image_height // 4) * (self.image_width // 4)),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.image_height // 4), (self.image_width) // 4)
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.image_height = 28
        self.image_width = 28
        self.input_dim = 1  # channels
        self.output_dim = 1  # output dimension

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128*(self.image_width//4)*(self.image_height//4), 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128*(self.image_height//4)*(self.image_width//4))
        x = self.fc(x)

        return x


class GAN(object):

    def __init__(self):
        self.epoch = 25
        self.sample_num = 16
        self.batch_size = 64
