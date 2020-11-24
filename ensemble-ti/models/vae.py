import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, infeatures, code_size=50):
        super(VAE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.relu = nn.ReLU()

        # Encoder architecture
        self.enc_fc1 = nn.Linear(self.infeatures, 64)
        self.enc_bn1 = nn.BatchNorm1d(64)

        # self.enc_fc2 = nn.Linear(512, 256)
        # self.enc_bn2 = nn.BatchNorm1d(256)

        # self.enc_fc3 = nn.Linear(256, 128)
        # self.enc_bn3 = nn.BatchNorm1d(128)

        self.enc_fc41 = nn.Linear(64, self.code_size)
        self.enc_fc42 = nn.Linear(64, self.code_size)
        self.dropout = nn.Dropout(0.1)

        # Decoder Architecture
        self.dec_fc1 = nn.Linear(self.code_size, 64)
        self.dec_bn1 = nn.BatchNorm1d(64)
        # self.dec_fc2 = nn.Linear(128, 256)
        # self.dec_bn2 = nn.BatchNorm1d(256)
        self.dec_fc3 = nn.Linear(64, self.infeatures)

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_fc1(x)))
        # x = self.relu(self.enc_bn2(self.enc_fc2(x)))
        # x = self.relu(self.enc_bn3(self.enc_fc3(x)))
        x = self.dropout(x)
        return self.enc_fc41(x), self.enc_fc42(x)

    def decode(self, z):
        x = self.relu(self.dec_bn1(self.dec_fc1(z)))
        # x = self.relu(self.dec_bn2(self.dec_fc2(x)))
        output = self.dec_fc3(x)
        return torch.sigmoid(output)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return z, decoder_out, mu, logvar


if __name__ == '__main__':
    vae = VAE(infeatures=892)
    input = torch.randn((32, 892))
    _, output, _, _ = vae(input)
    print(output.shape)
