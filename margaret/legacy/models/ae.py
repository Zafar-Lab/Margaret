import torch
import torch.nn as nn


class AE(nn.Module):
    # Implementation of a Regular autoencoder
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
        # Encoder
        z = self.encode(x)

        # Decoder
        out = self.decode(z)
        return out


class DAE(nn.Module):
    # Implementation of the Denoising Autoencoder
    def __init__(self, infeatures, code_size=10, noise_std=1.0):
        super(DAE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.relu = nn.ReLU()
        self.noise_std = noise_std

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
        # Add some noise to the input
        x = x + self.noise_std * torch.randn_like(x)

        # Encoder
        z = self.encode(x)

        # Decoder
        out = self.decode(z)
        return out


class SparseAE(nn.Module):
    # Implementation of the Sparse Autoencoder
    def __init__(self, infeatures, code_size=10, noise_std=1.0):
        super(SparseAE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.relu = nn.ReLU()
        self.noise_std = noise_std

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
        # Add some noise to the input
        x = x + self.noise_std * torch.randn_like(x)

        # Encoder
        z = self.encode(x)

        # Decoder
        out = self.decode(z)
        return z, out


class VAE(nn.Module):
    # Implementation of a Variational Autoencoder
    def __init__(self, infeatures, code_size=10):
        super(VAE, self).__init__()
        self.code_size = code_size
        self.infeatures = infeatures
        self.relu = nn.ReLU()

        # Encoder architecture
        self.enc_fc1 = nn.Linear(self.infeatures, 128)
        self.enc_bn1 = nn.BatchNorm1d(128)
        self.enc_fc2 = nn.Linear(128, 64)
        self.enc_bn2 = nn.BatchNorm1d(64)
        self.enc_fc31 = nn.Linear(64, self.code_size)
        self.enc_fc32 = nn.Linear(64, self.code_size)
        self.dropout = nn.Dropout(0.1)

        # Decoder Architecture
        self.dec_fc1 = nn.Linear(self.code_size, 64)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.dec_fc2 = nn.Linear(64, 128)
        self.dec_bn2 = nn.BatchNorm1d(128)
        self.dec_fc3 = nn.Linear(128, self.infeatures)

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_fc1(x)))
        x = self.relu(self.enc_bn2(self.enc_fc2(x)))
        x = self.dropout(x)
        return self.enc_fc31(x), self.enc_fc32(x)

    def decode(self, z):
        x = self.relu(self.dec_bn1(self.dec_fc1(z)))
        x = self.relu(self.dec_bn2(self.dec_fc2(x)))
        output = self.dec_fc3(x)
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        mu, logvar = self.encode(x)

        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoder_out = self.decode(z)
        return z, decoder_out, mu, logvar


if __name__ == "__main__":
    ae = AE(infeatures=100)
    dae = DAE(infeatures=100)
    vae = VAE(infeatures=100)
    input = torch.randn((32, 100))
    print(ae(input).shape)
    print(dae(input).shape)
    print(vae(input)[1].shape)
