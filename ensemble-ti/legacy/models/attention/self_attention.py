####################################################################
# Title: Self attention module
# Author: Kushagra Pandey
# Email: kp12@iitbbs.ac.in
####################################################################

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    SUPPORTED_MODES = ["dot", "embedded"]

    def __init__(self, mode="embedded", dropout=0):
        super(SelfAttention, self).__init__()
        if mode not in self.SUPPORTED_MODES:
            raise NotImplementedError(f"Only {self.SUPPORTED_MODES} are available!")

        self.mode = mode
        self.dropout = nn.Dropout(p=dropout)
        self.attn_map = []
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, k, q, v):
        # TODO: Mixture of Softmaxes
        b, ch, N = k.shape
        if self.mode == "dot":
            # Compute the attention using the dot product formulation
            self.attn_map = torch.matmul(q, k) / N
        elif self.mode == "embedded":
            self.attn_map = (ch ** -0.5) * torch.matmul(q, k)
            self.attn_map = self.softmax(self.attn_map)

        self.attn_map = self.dropout(self.attn_map)
        attended_value = torch.matmul(self.attn_map, v)
        attended_value = attended_value.permute(0, 2, 1).contiguous()
        return attended_value
