"""
Adversarial autoencoder
https://arxiv.org/abs/1511.05644
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Defining the global variables
n_epochs = 200
batch_size = 64
lr = 0.0002
b1, b2 = 0.5, 0.999
latent_dim = 100
img_size = 32
channels = 1
n_classes = 10
sample_interval = 500

os.makedirs('aae/images', exist_ok=True)

img_shape = (channels, img_size, img_size)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(FloatTensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z



# loss function
adversarial_loss = torch.nn.BCELoss()

# initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# configure data loader
os.makedirs('data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=True
)


# --------
# Training
# --------

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # for debug: getting a single batch of data
        # imgs, labels = next(iter(dataloader))

        batch_size = imgs.size(0)
        # adversarial ground truth
        valid = Variable(FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # configure input
        real_imgs = Variable(imgs.type(FloatTensor))

        # ----------------
        # Train generator
        # ----------------
        optimizer_G.zero_grad()

        # sample noise as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))

        # generate a batch of images
        gen_imgs = generator(z)

        # calculate loss
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # -------------------
        # Train discriminator
        # -------------------
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                                         d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:25], 'dcgan/images/%d.png' % batches_done, nrow=5, normalize=True)
