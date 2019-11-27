import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_speakers):
        super(Model, self).__init__()
        self.num_speakers = num_speakers

        self.conv1 = nn.Conv2d(1, 8, 4, 2)
        self.pool = nn.MaxPool2d(4, 1)
        self.linear1 = nn.Linear(8 * 11 * 96, 700)
        self.linear2 = nn.Linear(700, num_speakers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(-1, (8 * 11 * 96))

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.softmax(x)
        return x

