import torch
import torch.nn as nn

from models.attention.multi_head_attention import AttentionHead


class AttentionAE(nn.Module):
    def __init__(self, infeatures, code_size=10):
        super(AttentionAE, self).__init__()
        self.infeatures = infeatures
        self.code_size = code_size
        self.relu = nn.ReLU()

        # Encoder layers
        self.conv_1 = nn.Conv1d(1, 32, 3, padding=1, bias=False, stride=2)
        self.bn_1 = nn.BatchNorm1d(32)
        # self.dropout = nn.Dropout(0.1)

        self.conv_2 = nn.Conv1d(32, 64, 3, padding=1, bias=False, stride=2)
        self.bn_2 = nn.BatchNorm1d(64)

        self.mh1 = AttentionHead(64, 4, mh_dropout=0.1, att_dropout=0.1)
        self.mh2 = AttentionHead(64, 4, mh_dropout=0.1, att_dropout=0.1)

        self.fc1 = nn.Linear(512, code_size, bias=False)

        # Decoder Layers
        self.dec_fc1 = nn.Linear(code_size, 128, bias=False)
        self.dec_fc2 = nn.Linear(128, 256, bias=False)
        self.dec_fc3 = nn.Linear(256, self.infeatures, bias=False)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.mh1(x)
        x = self.mh2(x)

        x = torch.flatten(x, 1)
        code = self.relu(self.fc1(x))
        return code

    def decode(self, code):
        # Update the decoder to add more capacity
        x = self.relu(self.dec_fc1(code))
        x = self.relu(self.dec_fc2(x))
        x = self.relu(self.dec_fc3(x))
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out
