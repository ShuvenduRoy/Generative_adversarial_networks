import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np


class DCGAN():
    """Implementation of Deep convolutional GAN"""
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        optimizer = Adam(0.0002, 0.5)
