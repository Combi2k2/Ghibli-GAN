import torch
import torch.nn as nn

from autoencoder import Encoder, Decoder
from attention import SelfAttention
from layers import adaILN, ILN


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias, adaptive = True):
        super(ResnetBlock, self).__init__()
        
        self.adaptive = adaptive
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = use_bias)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = use_bias)
        
        if (adaptive):  self.norm1, self.norm2 = adaILN(dim), adaILN(dim)
        else:           self.norm1, self.norm2 = nn.InstanceNorm2d(dim), nn.InstanceNorm2d(dim)
        
        self.act = nn.ReLU(True)
    
    def forward(self, x, gamma = None, beta = None):
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta) if self.adaptive else self.norm1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta) if self.adaptive else self.norm1(out)
        
        return out + x

class Generator(nn.Module):
    def __init__(self, channels, features = 64, d_model = 528, n_downsampling = 3, n_blocks = 5):
        super().__init__()
        
        self.enc = Encoder(channels, d_model, features, n_downsampling, bias = False, act = "relu")
        self.dec = Decoder(channels, d_model, features, n_downsampling, bias = False, norm_layer = ILN, act = "relu")
        
        self.bottle_neck1 = nn.Sequential(*[ResnetBlock(d_model, False, adaptive = False) for _ in range(n_blocks)])
        self.bottle_neck2 = nn.ModuleList([ResnetBlock(d_model, False, adaptive = True) for _ in range(n_blocks)])
        
        self.attn_layer = SelfAttention(d_model, act = "relu")
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model, bias = False), nn.ReLU(True),
            nn.Linear(d_model, d_model, bias = False), nn.ReLU(True)
        )
        self.gamma = nn.Linear(d_model, d_model, bias = False)
        self.beta = nn.Linear(d_model, d_model, bias = False)

    def forward(self, input):
        x = self.enc(input)
        x = self.bottle_neck1(x)
        
        # Attention
        x, cam_logit, heat_map = self.attn_layer(x)
        
        # Apply adaptive layer
        x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_ = self.fc(x_.view(x_.shape[0], -1))
        
        gamma = self.gamma(x_)
        beta = self.beta(x_)

        for layer in self.bottle_neck2:
            x = layer(x, gamma, beta)
        
        x = self.dec(x)
        
        return x, cam_logit, heat_map

class Discriminator(nn.Module):
    def __init__(self, channels = 3, features = 64, d_model = 528, n_downsampling = 3):
    # def __init__(self, input_nc, ndf=64, n_layers=5):
        super().__init__()
        
        self.down = Encoder(channels, d_model, features, n_downsampling, bias = True, act = "leaky")
        self.attn = SelfAttention(d_model, act = "leaky")
        
        self.classifier = nn.Conv2d(d_model, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect', bias = False)

    def forward(self, input):
        x = self.down(input)
        x, cam_logit, heat_map = self.attn(x)
        
        return self.classifier(x), cam_logit, heat_map

if __name__ == '__main__':
    gen = Generator(channels = 3, features = 32, d_model = 200)
    disc = Discriminator(channels = 3, features = 32, d_model = 200)
    
    input = torch.randn(10, 3, 256, 256)
    
    g_img, g_cam, g_heatmap = gen(input)
    d_img, d_cam, d_heatmap = disc(input)
    
    print(g_img.shape, g_cam.shape, g_heatmap.shape)
    print(d_img.shape, d_cam.shape, d_heatmap.shape)