import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable
from einops import rearrange
from ..modules.conv import Conv, DWConv
from ..modules.block import *
__all__ = (
    "MSFF",
    "SPDConv",
)

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

class MSFF(nn.Module):    
    def __init__(self, dim, k1, k2, k3):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.conv = Conv(dim, dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, k1), padding=(0, k1 // 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (k1, 1), padding=(k1 // 2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, k2), padding=(0, k2 // 2), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (k2, 1), padding=(k2 // 2, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, k3), padding=(0, k3 // 2), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (k3, 1), padding=(k3 // 2, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.conv(self.avg_pool(x))

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + (attn_0 * attn_1 * attn_2)

        attn = self.act(self.conv3(attn))
        return attn * u


