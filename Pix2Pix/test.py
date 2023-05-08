import torch
import torch.nn as nn

m = nn.ConvTranspose2d(3, 10, 4, 2, 1)
x = torch.randn(10, 3, 1, 1)

y = m(x)
print(y.shape)