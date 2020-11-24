import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, infeatures, code_size=10):
        super(AE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.relu = nn.ReLU()

        # Encoder architecture
        self.enc_fc1 = nn.Linear(self.infeatures, 128)
        self.enc_bn1 = nn.BatchNorm1d(128)
        self.enc_fc2 = nn.Linear(128, 64)
        self.enc_bn2 = nn.BatchNorm1d(64)
        self.enc_fc3 = nn.Linear(64, self.code_size)
        self.enc_bn3 = nn.BatchNorm1d(self.code_size)

        # Decoder Architecture
        self.dec_fc1 = nn.Linear(self.code_size, 64)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.dec_fc2 = nn.Linear(64, 128)
        self.dec_bn2 = nn.BatchNorm1d(128)
        self.dec_fc3 = nn.Linear(128, self.infeatures)
        self.dec_bn3 = nn.BatchNorm1d(self.infeatures)

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_fc1(x)))
        x = self.relu(self.enc_bn2(self.enc_fc2(x)))
        x = self.relu(self.enc_bn3(self.enc_fc3(x)))
        return x

    def decode(self, z):
        x = self.relu(self.dec_bn1(self.dec_fc1(z)))
        x = self.relu(self.dec_bn2(self.dec_fc2(x)))
        output = self.relu(self.dec_bn3(self.dec_fc3(x)))
        return output

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out


if __name__ == '__main__':
    ae = AE(infeatures=100)
    input = torch.randn((32, 100))
    output = ae(input)
    print(output.shape)
