import torch
import torch.nn as nn
import torch.nn.functional as F 


class BpNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(BpNet, self).__init__()
        self.hidden1 = torch.nn.Linear(input_shape, 64)
        self.drop = torch.nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.hidden2 = torch.nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.hidden3 = torch.nn.Linear(32, 14)
        self.bn3 = nn.BatchNorm1d(14)
        self.predict = torch.nn.Linear(14, output_shape)

    def forward(self, x):
        x = self.drop(F.relu(self.hidden1(x)))
        x = self.bn1(x)
        x = self.drop(F.relu(self.hidden2(x)))
        x = self.bn2(x)
        x = F.relu(self.hidden3(x))
        x = self.bn3(x)
        x = F.relu(self.predict(x))
        return x
