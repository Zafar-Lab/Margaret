import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(self, infeatures, code_size=50):
        super(DAE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.relu = nn.ReLU()

        # Encoder architecture
        self.enc_fc1 = nn.Linear(self.infeatures, 512)
        self.enc_bn1 = nn.BatchNorm1d(512)

        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_bn2 = nn.BatchNorm1d(256)

        self.enc_fc3 = nn.Linear(256, 128)
        self.enc_bn3 = nn.BatchNorm1d(128)

        self.enc_fc4 = nn.Linear(128, self.code_size)
        self.dropout = nn.Dropout(0.1)

        # Decoder Architecture
        self.dec_fc1 = nn.Linear(self.code_size, 128)
        self.dec_bn1 = nn.BatchNorm1d(128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_bn2 = nn.BatchNorm1d(256)
        self.dec_fc3 = nn.Linear(256, 512)
        self.dec_bn3 = nn.BatchNorm1d(512)
        self.dec_fc4 = nn.Linear(512, self.infeatures)

    def encode(self, x):
        # add noise to the data
        x = x + torch.randn_like(x)
        x = self.relu(self.enc_bn1(self.enc_fc1(x)))
        x = self.relu(self.enc_bn2(self.enc_fc2(x)))
        x = self.relu(self.enc_bn3(self.enc_fc3(x)))
        x = self.dropout(x)
        return self.enc_fc4(x)

    def decode(self, z):
        x = self.relu(self.dec_bn1(self.dec_fc1(z)))
        x = self.relu(self.dec_bn2(self.dec_fc2(x)))
        x = self.relu(self.dec_bn3(self.dec_fc3(x)))
        output = self.dec_fc4(x)
        return torch.sigmoid(output)

    def forward(self, x):
        z = self.encode(x)
        decoder_out = self.decode(z)
        return decoder_out


if __name__ == '__main__':
    dae = DAE(infeatures=892)
    input = torch.randn((32, 892))
    _, output, _, _ = vae(input)
    print(output.shape)
