import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.main(x)
    
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]
        # Initial Conv
        out_features = 64
        model = [
            nn.Conv2d(channels, out_features, kernel_size = 7, padding = 3, padding_mode = 'reflect'),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        model += [
            nn.Conv2d(out_features, out_features*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features*2, out_features*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 4),
            nn.ReLU(inplace=True),
        ]

        # Res blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features*4)]
        
        # Upsampling
        model += [
            nn.ConvTranspose2d(out_features*4, out_features*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_features*2, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        # Output layer
        model += [
            nn.Conv2d(out_features, channels, kernel_size = 7, padding = 3),
            nn.Tanh()
        ]
        
        # Unpacking
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        self.output_shape = (1, height//16, width//16)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride = 2, padding = 1)]
            if normalize:
                layers += [nn.InstanceNorm2d(out_filters)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding = 1, padding_mode='zeros')
        )
    
    def forward(self, img):
        return self.model(img)
    
if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256)

    gen = GeneratorResNet((3, 256, 256), 9)
    disc = Discriminator((3, 256, 256))

    from utils import count_parameters

    print(count_parameters(gen))
    print(count_parameters(disc))

    print(gen(input).shape)
    print(disc(input).shape)