import torch
import torch.nn as nn

from models.layers import Conv1d
from models.layers import SqueezeExcitation
from models.attention.multi_head_attention import MultiHeadAttention


# A CNN variant for the multilabel drug resistance prediction problem
class DeepCNNResistancePredictor_ws(nn.Module):
    def __init__(self, infeatures, num_classes):
        super(DeepCNNResistancePredictor_ws, self).__init__()
        self.infeatures = infeatures
        self.num_classes = num_classes

        # Network layers
        self.conv_1 = Conv1d(1, 64, 3, padding=1, bias=False)
        self.bn_1 = nn.GroupNorm(32, 64)
        # Using dropout 0.3 gives an overall best performance of 94.5 AUC as compared to 94.4 in this case
        self.dropout = nn.Dropout2d(0.3)
        self.maxpool = nn.MaxPool1d(2)

        self.conv_2 = Conv1d(64, 128, 3, padding=1, bias=False, stride=2)
        self.bn_2 = nn.GroupNorm(32, 128)

        self.conv_3 = Conv1d(128, 256, 3, padding=2, dilation=2, bias=False, stride=2)
        self.bn_3 = nn.GroupNorm(32, 256)

        self.conv_4 = Conv1d(256, 512, 3, dilation=2, padding=2, bias=False, stride=2)
        self.bn_4 = nn.GroupNorm(32, 512)

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


# A CNN variant for the multilabel drug resistance prediction problem
class DeepCNNResistancePredictor_mh(nn.Module):
    def __init__(self, infeatures, num_classes):
        super(DeepCNNResistancePredictor_mh, self).__init__()
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
        self.mh = MultiHeadAttention(512, 4, share_weights=True, dropout=0.3)

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
        x_att = self.mh(x)
        x = x + x_att
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.classifier(x)

        return x


# A CNN variant for the multilabel drug resistance prediction problem
class DeepCNNResistancePredictor_mh2(nn.Module):
    def __init__(self, infeatures, num_classes):
        super(DeepCNNResistancePredictor_mh2, self).__init__()
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

        # self.conv_4 = nn.Conv1d(256, 512, 3, dilation=2, padding=2, bias=False, stride=2)
        # self.bn_4 = nn.BatchNorm1d(512)
        self.mh1 = MultiHeadAttention(256, 4, share_weights=True, dropout=0.3)
        self.conv1 = nn.Conv1d(256, 256, 1, bias=False)
        self.norm1a = nn.GroupNorm(256, 256)
        self.norm1b = nn.GroupNorm(256, 256)

        self.mh2 = MultiHeadAttention(256, 4, share_weights=True, dropout=0.3)
        self.conv2 = nn.Conv1d(256, 256, 1, bias=False)
        self.norm2a = nn.GroupNorm(256, 256)
        self.norm2b = nn.GroupNorm(256, 256)

        self.mh3 = MultiHeadAttention(256, 4, share_weights=True, dropout=0.3)
        self.conv3 = nn.Conv1d(256, 256, 1, bias=False)
        self.norm3a = nn.GroupNorm(256, 256)
        self.norm3b = nn.GroupNorm(256, 256)

        self.fc1 = nn.Linear(256, 256, bias=False)
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

        x_att = self.mh1(x)
        x = self.norm1a(x + x_att)
        x = self.relu(self.norm1b(x + self.conv1(x)))

        x_att = self.mh2(x)
        x = self.norm2a(x + x_att)
        x = self.relu(self.norm2b(x + self.conv1(x)))

        x_att = self.mh3(x)
        x = self.norm3a(x + x_att)
        x = self.relu(self.norm3b(x + self.conv1(x)))

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.classifier(x)

        return x


# A CNN variant for the multilabel drug resistance prediction problem
class DeepCNNResistancePredictor_se(nn.Module):
    def __init__(self, infeatures, num_classes):
        super(DeepCNNResistancePredictor_se, self).__init__()
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

        self.se1 = SqueezeExcitation(128)
        self.se2 = SqueezeExcitation(256)
        self.se3 = SqueezeExcitation(512)

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
        # x = self.se1(x)

        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.dropout(x)
        x = self.se2(x)

        x = self.relu(self.bn_4(self.conv_4(x)))
        x = self.dropout(x)
        x = self.se3(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.classifier(x)

        return x
