import torch.nn as nn
import torch

from .attention_utils import PadBlock
from .attention_utils import LocalPermuteModule

class PositionEmbeddding2D(nn.Module):
    """
    2D position encoding, borrowed from DETR PositionEmbeddingSine class
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, temperature = 10000, normalize = False, scale = None, device = 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, N: int, E: int, H: int, W: int):
        """
        Args:
            N for batch size, E for embedding size (channel of feature), H for height, W for width
        Returns:
            pos_embed: positional encoding with shape (N, E, H, W)
        """
        assert E % 2 == 0, "Embedding size should be even number"

        y_embed = torch.ones(N, H, W, dtype = torch.float32, device = self.device).cumsum(dim = 1)
        x_embed = torch.ones(N, H, W, dtype = torch.float32, device = self.device).cumsum(dim = 2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(E//2, dtype=torch.float32, device=self.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode = 'floor') / (E//2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos_embed.requires_grad_(False)

        return pos_embed

class SpatialMultiheadAttention(nn.Module):
    """
    Modified based on https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    local spatial window attention with absolute positional encoding, i.e. based the standard nn.MultiheadAttention module
    Args:
        embed_dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embed_dim, num_heads, window_size = 7, dropout = 0.2):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout)
        
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)
        
        pos2d = PositionEmbeddding2D()
        
        pos_embed = pos2d(N = 1, E = embed_dim, H = window_size, W = window_size)
        pos_embed = pos_embed[0, ...].permute(1, 2, 0).flatten(0, 1)[:, None, :]
        
        self.register_buffer('spatial_pos', pos_embed)
        self._reset_parameters()

    def forward(self, x):
        """
        x: (N, H, W, C)
        value: value should be None for encoder self-attention, value is not None for the Transformer decoder self-attention
        local_pos_embed: (window_size, window_size, C)
        return:
           (N, H, W, C)
        """
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        x_permute = self.permute_helper.permute(x_pad, x_pad.size()) #(window_size*window_size, N*H/window_size*W/window_size, C)
        
        k = q = x_permute + self.spatial_pos
        value = x_permute

        out = self.attn(q, k, value)[0]
        out = self.permute_helper.rev_permute(out, x_pad.size()) #(N*T, H, W, C)
        out = self.pad_helper.depad_if_needed(out, x.size())
        
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

if __name__ == '__main__':
    model = SpatialMultiheadAttention(128, 8)
    
    inputs = torch.randn(1, 16, 16, 128)
    output = model(inputs)
    
    print(output.shape)
    