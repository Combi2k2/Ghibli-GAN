"""
Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and therefor
it should be called critic)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, act = "relu", use_dropout = False):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out) if self.use_dropout else out
        
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, act = "relu", use_dropout = False):
        super(Up, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out) if self.use_dropout else out
        
        return out

class Generator(nn.Module):
    def __init__(self, in_channels = 3, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        # input = N x 3 x 256 x 256
        self.down1 = Down(features, features * 2, act = "Leaky")
        self.down2 = Down(features * 2, features * 4, act = "Leaky")
        self.down3 = Down(features * 4, features * 8, act = "Leaky")
        self.down4 = Down(features * 8, features * 8, act = "Leaky")
        self.down5 = Down(features * 8, features * 8, act = "Leaky")
        self.down6 = Down(features * 8, features * 8, act = "Leaky")
        
        self.bottle_neck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode = "reflect"),
            nn.ReLU()
        )
        
        self.up1 = Up(features * 8, features * 8, act = "relu", use_dropout = True)
        self.up2 = Up(features * 8 * 2, features * 8, act="relu", use_dropout = True)
        self.up3 = Up(features * 8 * 2, features * 8, act="relu", use_dropout = True)
        self.up4 = Up(features * 8 * 2, features * 8, act="relu", use_dropout = False)
        self.up5 = Up(features * 8 * 2, features * 4, act="relu", use_dropout = False)
        self.up6 = Up(features * 4 * 2, features * 2, act="relu", use_dropout = False)
        self.up7 = Up(features * 2 * 2, features, act="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottle_neck = self.bottle_neck(d7)
        
        up1 = self.up1(bottle_neck)
        up2 = self.up2(torch.cat([up1, d7], dim = 1))
        up3 = self.up3(torch.cat([up2, d6], dim = 1))
        up4 = self.up4(torch.cat([up3, d5], dim = 1))
        up5 = self.up5(torch.cat([up4, d4], dim = 1))
        up6 = self.up6(torch.cat([up5, d3], dim = 1))
        up7 = self.up7(torch.cat([up6, d2], dim = 1))
        
        return self.final_up(torch.cat([up7, d1], dim = 1))

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 128, 256, 512]):
        super().__init__()

        in_channels *= 2
        layers = []
        
        for feature in features:
            layers.append(nn.Conv2d(in_channels, feature, 4, stride = 1 if feature == features[-1] else 2, padding = 1, padding_mode = "reflect"))
            layers.append(nn.BatchNorm2d(feature) if feature == features[0] else nn.Identity())
            layers.append(nn.LeakyReLU(0.2))
            
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim = 1))

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    
    gen = Generator()
    disc = Discriminator()
    
    out_gen = gen(x)
    out_disc = disc(x, y)
    
    print(out_gen.shape)
    print(out_disc.shape)