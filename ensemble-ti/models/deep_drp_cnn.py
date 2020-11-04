import torch
import torch.nn as nn


# A CNN variant for the multilabel drug resistance prediction problem
class DeepCNNResistancePredictor(nn.Module):
    def __init__(self, infeatures, num_classes):
        super(DeepCNNResistancePredictor, self).__init__()
        self.infeatures = infeatures
        self.num_classes = num_classes

        # Network layers
        self.conv_1 = nn.Conv1d(1, 64, 3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm1d(64)
        # Using dropout 0.3 gives an overall best performance of 94.5 AUC as compared to 94.4 in this case
        self.dropout = nn.Dropout2d(0.3)
        self.maxpool = nn.MaxPool1d(2)

        self.conv_2 = nn.Conv1d(64, 128, 3, padding=1, bias=False, stride=2)
        self.bn_2 = nn.BatchNorm1d(128)

        self.conv_3 = nn.Conv1d(128, 256, 3, padding=2, dilation=2, bias=False, stride=2)
        self.bn_3 = nn.BatchNorm1d(256)

        self.conv_4 = nn.Conv1d(256, 512, 3, dilation=2, padding=2, bias=False, stride=2)
        self.bn_4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256, bias=False)
        self.classifier = nn.Linear(256, self.num_classes, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU6()

    def forward(self, x):
        identity = x
        x = x.unsqueeze(1)
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn_2(self.conv_2(x)))

        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.dropout(x)

        x = self.relu(self.bn_4(self.conv_4(x)))
        x = self.dropout(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.classifier(x)

        return x
