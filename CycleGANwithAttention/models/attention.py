import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, feat_dim, act = 'relu'):
        super().__init__()
        self.gap_fc = nn.Linear(feat_dim, 1)
        self.gmp_fc = nn.Linear(feat_dim, 1)
        
        self.conv1x1 = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size = 1)
        self.relu = nn.ReLU(True) if act == 'relu' else nn.LeakyReLU(0.2)
    
    def forward(self, x):
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)
        
        return x, cam_logit, heatmap

if __name__ == '__main__':
    model = SelfAttention(123)
    
    inputs = torch.randn(1, 123, 32, 32)
    output, cam, heat = model(inputs)
    
    print(output.shape)
    print(cam.shape)
    print(heat.shape)
    