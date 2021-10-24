import torch
import torch.nn as nn

from models.attention.self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self, inplanes, num_heads, dropout=0, mode="embedded", share_weights=False
    ):
        super(MultiHeadAttention, self).__init__()

        self.inplanes = inplanes
        self.num_heads = num_heads
        self.mode = mode
        self.share_weights = share_weights
        self.truncated_planes = self.inplanes // self.num_heads

        self.key_transform = nn.Conv1d(
            self.inplanes, self.truncated_planes * self.num_heads, 1, bias=False
        )
        self.query_transform = nn.Conv1d(
            self.inplanes, self.truncated_planes * self.num_heads, 1, bias=False
        )
        self.value_transform = nn.Conv1d(
            self.inplanes, self.truncated_planes * self.num_heads, 1, bias=False
        )

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


class AttentionHead(nn.Module):
    def __init__(self, inplanes, num_heads, mh_dropout=0, att_dropout=0):
        super(AttentionHead, self).__init__()
        self.inplanes = inplanes
        self.num_heads = num_heads
        self.mh_dropout = mh_dropout
        self.dropout = nn.Dropout(att_dropout)

        self.mh_module = nn.Sequential(
            nn.GroupNorm(1, self.inplanes),
            MultiHeadAttention(self.inplanes, self.num_heads, dropout=self.mh_dropout),
        )
        self.ff_module = nn.Sequential(
            nn.GroupNorm(1, self.inplanes), nn.Conv1d(self.inplanes, self.inplanes, 1)
        )

    def forward(self, x):
        # Attention module
        identity = x

        x = identity + self.dropout(self.mh_module(x))

        # FF Module
        identity = x
        x = identity + self.dropout(self.ff_module(x))
        return x
