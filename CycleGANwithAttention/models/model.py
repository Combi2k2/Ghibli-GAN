import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

from .autoencoder import Encoder, Decoder
from .attention import SpatialMultiheadAttention
from .attention_utils import DropPath

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = use_bias),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x) + x

class ConvFFN(nn.Module):
    def __init__(self, dim, bias = True, dropout = 0.2, norm = nn.utils.weight_norm):
        super().__init__()
        self.conv = nn.Sequential(
            norm(nn.Conv2d(dim, dim, kernel_size = 1, bias = bias)),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            
            norm(nn.Conv2d(dim, dim, kernel_size = 3, padding = 1, groups = dim, bias = bias)),    # deepwise
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            norm(nn.Conv2d(dim, dim, kernel_size = 1, bias = bias)),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(dropout)
        )
    
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, channels = 3, features = 64, d_model = 528, n_downsampling = 3, n_blocks = 5, nhead = 8, window_size = 7, dropout = 0.2):
        super().__init__()
        
        self.embed_dim = d_model
        
        self.enc = Encoder(channels, d_model, features = features, n_downsampling = n_downsampling, bias = False, act = "leaky")
        self.dec = Decoder(channels, d_model, features = features, n_downsampling = n_downsampling, bias = False, act = "relu")
        
        self.bottleneck = nn.Sequential(*[ResnetBlock(d_model, False) for _ in range(n_blocks)])
        
        self.spatialAttention = SpatialMultiheadAttention(d_model, num_heads = nhead, window_size = window_size, dropout = dropout)
        self.spatialConvFFN = ConvFFN(d_model, bias = False, dropout = dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.drop_path = DropPath(dropout)

    def forward(self, input):
        x = self.enc(input)
        x = self.bottleneck(x)
        # Attention
        x = x.permute(0, 2, 3, 1)   # N, H, W, C
        x = x + self.drop_path(self.spatialAttention(self.norm1(x)))
        x = x.permute(0, 3, 1, 2)   # N, C, H, C
        
        gap_logit = torch.nn.functional.adaptive_avg_pool2d(x[:, :self.embed_dim // 2, :, :], 1)
        gmp_logit = torch.nn.functional.adaptive_max_pool2d(x[:, self.embed_dim // 2:, :, :], 1)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], dim = 1)
        
        x = x + self.drop_path(self.spatialConvFFN(self.norm2(x)))
        
        heatmap = torch.sum(x, dim = 1, keepdim = True)
        output = self.dec(x)
        
        return output, cam_logit, heatmap

class Discriminator(nn.Module):
    def __init__(self, channels = 3, features = 64, d_model = 528, n_downsampling = 3, n_blocks = 5, nhead = 8, window_size = 7, dropout = 0.2):
    # def __init__(self, input_nc, ndf=64, n_layers=5):
        super().__init__()
        
        self.embed_dim = d_model
        moduleList = []
        
        for i in range(n_downsampling):
            in_channels = features << (i - 1) if i > 0 else channels
            out_channels = features << i if i < n_downsampling - 1 else d_model
            
            moduleList.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect')))
            moduleList.append(nn.LeakyReLU(0.2))
        
        
        self.down = nn.Sequential(*moduleList)
        self.bottleneck = nn.Sequential(*[ResnetBlock(d_model, True) for _ in range(n_blocks)])
        
        self.spatialAttention = SpatialMultiheadAttention(d_model, num_heads = nhead, window_size = window_size, dropout = dropout)
        self.spatialConvFFN = ConvFFN(d_model, bias = True, dropout = dropout, norm = nn.utils.spectral_norm)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.drop_path = DropPath(dropout)

        self.classifier = nn.utils.spectral_norm(nn.Conv2d(d_model, 1, kernel_size = 4, stride = 1, padding = 1, bias = False))

    def forward(self, input):
        x = self.down(input)
        x = self.bottleneck(x)
        # Attention
        x = x.permute(0, 2, 3, 1)   # N, H, W, C
        x = x + self.drop_path(self.spatialAttention(self.norm1(x)))
        x = x.permute(0, 3, 1, 2)   # N, C, H, C
        
        gap_logit = torch.nn.functional.adaptive_avg_pool2d(x[:, :self.embed_dim // 2, :, :], 1)
        gmp_logit = torch.nn.functional.adaptive_max_pool2d(x[:, self.embed_dim // 2:, :, :], 1)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], dim = 1)
        
        x = x + self.drop_path(self.spatialConvFFN(self.norm2(x)))
        
        heatmap = torch.sum(x, dim = 1, keepdim = True)
        output = self.classifier(x)

        return output, cam_logit, heatmap

if __name__ == '__main__':
    gen = Generator(channels = 3, features = 32, d_model = 200)
    disc = Discriminator(channels = 3, features = 32, d_model = 200)
    
    input = torch.randn(10, 3, 256, 256)
    
    g_img, g_cam, g_heatmap = gen(input)
    d_img, d_cam, d_heatmap = disc(g_img)
    
    print(g_img.shape, g_cam.shape, g_heatmap.shape)
    print(d_img.shape, d_cam.shape, d_heatmap.shape)