import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)  # (B*num_windows, N, C)

        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B*, num_heads, N, dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)

        out = self.proj(out)
        out = out.view(B, H // self.window_size, W // self.window_size,
                       self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H, W, C)
        return out


class MultiScaleWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_sizes=[2, 4, 8]):
        super(MultiScaleWindowAttention, self).__init__()
        self.layers = nn.ModuleList([
            WindowAttention(dim, w_size, num_heads) for w_size in window_sizes
        ])

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        outs = []
        for attn in self.layers:
            outs.append(attn(x))
        out = sum(outs) / len(outs)  # mean fusion
        out = out.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H, W]
        return out

msa = MultiScaleWindowAttention(dim=96, num_heads=4)
x = torch.randn(8, 96, 64, 64)  # [B, C, H, W]
out = msa(x)
print(out.shape)  # -> torch.Size([8, 96, 64, 64])

