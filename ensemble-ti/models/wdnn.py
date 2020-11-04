import torch
import torch.nn as nn
import torch.nn.functional as F

# Implements the WDNN classifier that is implemented in the paper
# "Beyond multidrug resistance: Leveraging rare variants with machine and
# statistical learning models in Mycobacterium tuberculosis resistance prediction"
class WDNNResistancePredictor(nn.Module):
    def __init__(self, infeatures, num_classes, nodes=[256, 256, 256]):
        super(WDNNResistancePredictor, self).__init__()
        self.num_classes = num_classes
        self.infeatures = infeatures
        self.nodes = nodes

        # Network layers
        self.linear_1 = nn.Linear(self.infeatures, self.nodes[0], bias=False)
        self.bn_1 = nn.BatchNorm1d(self.nodes[0])

        self.linear_2 = nn.Linear(self.nodes[0], self.nodes[1], bias=False)
        self.bn_2 = nn.BatchNorm1d(self.nodes[1])

        self.linear_3 = nn.Linear(self.nodes[1], self.nodes[2], bias=False)
        self.bn_3 = nn.BatchNorm1d(self.nodes[2])

        outplanes = self.nodes[-1] + self.infeatures
        self.classifier = nn.Linear(outplanes, self.num_classes)
        # Also used a model with dropout set to 0.5
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn_1(self.linear_1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_2(self.linear_2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_3(self.linear_3(x)))
        x = self.dropout(x)

        # Concat
        x = torch.cat([identity, x], dim=-1)
        out = self.classifier(x)
        return out