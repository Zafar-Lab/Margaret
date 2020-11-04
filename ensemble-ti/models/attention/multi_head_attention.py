####################################################################
# Title: MultiHead attention module
# Author: Kushagra Pandey
# Email: kp12@iitbbs.ac.in
####################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.layers as L
from .self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, inplanes, num_heads, dropout=0, mode='embedded', share_weights=False):
        super(MultiHeadAttention, self).__init__()

        self.inplanes = inplanes
        self.num_heads = num_heads
        self.mode = mode
        self.share_weights = share_weights
        self.truncated_planes = self.inplanes // self.num_heads
        
        self.key_transform = nn.Conv1d(self.inplanes, self.truncated_planes * self.num_heads, 1, bias=False)
        self.query_transform = nn.Conv1d(self.inplanes, self.truncated_planes * self.num_heads, 1, bias=False)
        self.value_transform = nn.Conv1d(self.inplanes, self.truncated_planes * self.num_heads, 1, bias=False)

        if self.share_weights:
            # Make the query and the key weights shared
            self.query_transform = self.key_transform
        self.self_attn = SelfAttention(mode=self.mode, dropout=dropout)
        self.conv = nn.Conv1d(self.inplanes, self.inplanes, 1, bias=False)

    def forward(self, x):
        identity = x

        key = self.key_transform(x)
        query = self.query_transform(x)
        value = self.value_transform(x)

        # Dimensions of Value before projection to a low dim space
        b, ch, n = value.shape

        assert key.shape[1] == query.shape[1]

        b, ch, _ = key.shape
        k = key.view(b * self.num_heads, self.truncated_planes, -1)
        q = query.view(b * self.num_heads, self.truncated_planes, -1).permute(0, 2, 1)
        v = value.view(b * self.num_heads, self.truncated_planes, -1).permute(0, 2, 1)

        x = self.self_attn(k, q, v)
        x = x.view(b, self.truncated_planes * self.num_heads, n)

        # Apply a linear transformation
        x = self.conv(x)
        return x
