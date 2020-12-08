import torch
import torch.nn as nn


class MetricEncoder(nn.Module):
    def __init__(self, infeatures, code_size=10):
        super(MetricEncoder, self).__init__()
        self.infeatures = infeatures
        self.code_size = code_size

        # Encoder architecture
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Encoder Architecture
        self.fc1 = nn.Linear(self.infeatures, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, self.code_size)
        self.bn3 = nn.BatchNorm1d(self.code_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(self.fc3(x))
        return x


if __name__ == '__main__':
    encoder = MetricEncoder(infeatures=100)
    input = torch.randn((32, 100))
    output = encoder(input)
    print(output.shape)
