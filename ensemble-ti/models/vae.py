import torch
import torch.nn as nn
import torch.nn.functional as F


# Credits: Code shamelessly taken from https://github.com/pytorch/examples/tree/master/vae
class VAE(nn.Module):
    def __init__(self, infeatures, code_size=50):
        super(VAE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.fc1 = nn.Linear(self.infeatures, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc41 = nn.Linear(128, self.code_size)
        self.fc42 = nn.Linear(128, self.code_size)

        self.fc5 = nn.Linear(self.code_size, 256)
        self.fc6 = nn.Linear(256, self.infeatures)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc41(h3), self.fc42(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        kl = self.compute_kl(mu, logvar)
        return z, decoder_out, kl

    def decode(self, z):
        h3 = F.relu(self.fc5(z))
        return self.fc6(h3)

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
