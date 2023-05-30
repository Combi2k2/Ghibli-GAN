from torch.nn.parameter import Parameter
from torch import nn

import torch

class adaILN(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.ones(1, num_features, 1, 1) * 0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim = [2, 3], keepdim = True), torch.var(input, dim = [2, 3], keepdim=True)
        ln_mean, ln_var = torch.mean(input, dim = [1, 2, 3], keepdim = True), torch.var(input, dim = [1, 2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out

class ILN(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        
        self.rho = Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim = [2, 3], keepdim = True), torch.var(input, dim = [2, 3], keepdim = True)
        ln_mean, ln_var = torch.mean(input, dim = [1, 2, 3], keepdim = True), torch.var(input, dim = [1, 2, 3], keepdim = True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out