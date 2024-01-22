import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils

from torch.utils.data import Dataset
from PIL import Image

from model import UnetGenerator, Discriminator


def calculate_ssim(img1, img2):
    # Calculate SSIM using PyTorch's functional module
    ssim_value = SSIM()(img1, img2)

    return ssim_value.item()


def calculate_rmse(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


generator = UnetGenerator(3, 1, 8, 64, use_dropout=True).cuda()
ckpt = torch.load('cpt/last.pt')

generator.load_state_dict(ckpt["g"])
generator.eval()

path = 'IMG/KITTI/000025_10.png'
save = 'sam/000025_10.png'

names = os.listdir(path)

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
for name in names:
    xxx = os.path.join(path, name)
    yyy = os.path.join(save, name)

    xxx_img = Image.open(xxx).convert('RGB')
    w, h = xxx_img.size

    xxx_img = transform(xxx_img).unsqueeze(0).cuda()
    yyy_img = generator(xxx_img)
    yyy_img = transforms.Resize((h, w))(yyy_img)

    utils.save_image(
        yyy_img,
        yyy,
        normalize=True,
        range=(-1, 1),
    )
