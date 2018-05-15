"""
StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation
http://arxiv.org/abs/1711.09020
"""

import argparse
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from star_gan.datasets import *
from star_gan.models import *

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--dataset_name', type=str, default="img_align_celeba", help='name of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image samples')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--residual_blocks', type=int, default=6, help='number of residual blocks in generator')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
opt = parser.parse_args()
print(opt)

c_dim = len(opt.selected_attrs)
img_shape = (opt.channels, opt.img_height, opt.img_width)
cuda = True if torch.cuda.is_available() else False

# Loss function : cycle loss function
criterion_cycle = torch.nn.L1Loss()


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

# Initialize generator and discriminator
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks, c_dim=c_dim)
discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_cycle.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth' % opt.epoch))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth' % opt.epoch))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Configure dataloaders
train_transforms = [transforms.Resize(int(1.12*opt.img_height), Image.BICUBIC),
                    transforms.RandomCrop(opt.img_height),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(CelebADataset("../../data/%s" % opt.dataset_name, transforms_=train_transforms, mode='train', attributes=opt.selected_attrs),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_transforms = [  transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

val_dataloader = DataLoader(CelebADataset("../../data/%s" % opt.dataset_name, transforms_=val_transforms, mode='val', attributes=opt.selected_attrs),
                        batch_size=10, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

