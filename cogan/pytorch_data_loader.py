import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, folder_path, transforms_=None):
        self.transform = transforms_
        self.files = sorted(glob.glob('%s/*.*' % folder_path))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)
