import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_speakers):
        super(Model, self).__init__()
        self.num_speakers = num_speakers
        self.lin_size = 16 * 13 * 98

        self.conv1 = nn.Conv2d(1, 64, 4, 2)
        self.conv2 = nn.Conv2d(64, 32, 2, 1)
        self.conv3 = nn.Conv2d(32, 16, 1, 1)
        self.pool = nn.MaxPool2d(1, 1)
        self.linear1 = nn.Linear(self.lin_size, 700)
        self.linear2 = nn.Linear(700, num_speakers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x)
        x = x.view(-1, (self.lin_size))

        x = F.relu(self.linear1(x))
        embedding = x

        x = F.relu(self.linear2(x))
        x = F.softmax(x)
        return x, embedding

