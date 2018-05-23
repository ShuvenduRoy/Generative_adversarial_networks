"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
http://arxiv.org/abs/1609.04802
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from srgan.models import *
from srgan.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)


epoch = 0
n_epochs = 200
dataset_name = 'img_align_celeba'
batch_size = 1
lr = 0.0002
b1, b2 = 0.5, 0.999
latent_dim = 10
decay_epoch = 100
hr_height = 256
hr_width = 256
channels = 3
sample_interval = 1000
checkpoint_interval = -1
n_cpu = 8

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(hr_height / 2 ** 4), int(hr_width / 2 ** 4)
patch = (batch_size, 1, patch_h, patch_w)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator()
feature_extractor = FeatureExtractor()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_lr = Tensor(batch_size, channels, hr_height // 4, hr_width // 4)
input_hr = Tensor(batch_size, channels, hr_height, hr_width)
# Adversarial ground truths
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

# Transforms for low resolution images and high resolution images
lr_transforms = [transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

hr_transforms = [transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

dataloader = DataLoader(
    ImageDataset("E:\\Datasets\\%s" % dataset_name, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
    batch_size=batch_size, shuffle=True, num_workers=n_cpu)

# ----------
#  Training
# ----------

for epoch in range(epoch, n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(input_lr.copy_(imgs['lr']))
        imgs_hr = Variable(input_hr.copy_(imgs['hr']))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        gen_validity = discriminator(gen_hr)
        loss_GAN = criterion_GAN(gen_validity, valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = Variable(feature_extractor(imgs_hr).data, requires_grad=False)
        loss_content = criterion_content(gen_features, real_features)

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, n_epochs, i, len(dataloader),
               loss_D.item(), loss_G.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            # Save image sample
            save_image(torch.cat((gen_hr.data, imgs_hr.data), -2),
                       'images/%d.png' % batches_done, normalize=True)

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
