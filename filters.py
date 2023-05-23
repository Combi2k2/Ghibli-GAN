import torch
import torch.nn as nn
import numpy as np

def box_filter(x, r):
    channels = x.shape[1]
    weight = 1 / ((2 * r + 1)**2)
    
    box_kernel = nn.Conv2d(
        in_channels = channels,
        out_channels = channels,
        kernel_size = 2 * r + 1,
        padding = r,
        padding_mode = 'reflect',
        bias = False,
        groups = channels
    )
    box_kernel.weight = nn.Parameter(torch.ones_like(box_kernel.weight) / weight)
    box_kernel.requires_grad_ = False
    
    return box_kernel(x)

def guided_filter(x, y, r, eps = 1e-2):
    '''
        Input:
            x, y:   tensor or np.ndarray of format N x C x H x W
            r:      an interger
        Return:
            An extracted surface representation with textures and details removed
            The extracted representation has the same shape with the inputs

    '''
    _, _, H, W = x.shape
    
    N = box_filter(torch.ones((1, 1, H, W), dtype = x.dtype), r)
    
    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output

def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = torch.split(image1, [1, 1, 1], dim = 1)
    b2, g2, r2 = torch.split(image2, [1, 1, 1], dim = 1)
    
    if mode == 'normal':
        b_weight = torch.normal(mean = 0.114, std = 0.1, size = (1,))
        g_weight = torch.normal(mean = 0.587, std = 0.1, size = (1,))
        r_weight = torch.normal(mean = 0.299, std = 0.1, size = (1,))
    elif mode == 'uniform':
        b_weight = torch.from_numpy(np.random.uniform(low = 0.014, high = 0.214, size = (1,))).float()
        g_weight = torch.from_numpy(np.random.uniform(low = 0.487, high = 0.687, size = (1,))).float()
        r_weight = torch.from_numpy(np.random.uniform(low = 0.199, high = 0.399, size = (1,))).float()
    
    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    output2 = (b_weight * b2 + g_weight * g2 + r_weight * r2) / (b_weight + g_weight + r_weight)
    
    return output1, output2

if __name__ == '__main__':
    x = torch.randn((16, 3, 64, 64))
    y = torch.randn((16, 3, 64, 64))
    
    print(guided_filter(x, y, 2).shape)