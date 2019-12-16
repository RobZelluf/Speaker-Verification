import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, num_speakers, embedding_dim=700):
        super(Model, self).__init__()
        self.num_speakers = num_speakers
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.lin_size = 32 * 93
        print("Embedding_dim:", embedding_dim)

        self.conv1 = nn.Conv1d(input_dim[0], 128, 8, 2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 4, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, 2, 1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool = nn.AvgPool1d(1, 1)
        self.linear1 = nn.Linear(self.lin_size, 700)
        self.bn4 = nn.BatchNorm1d(700)
        self.linear2 = nn.Linear(700, num_speakers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = self.pool(x)
        x = x.view(-1, self.lin_size)

        x = F.relu(self.linear1(x))
        embedding = x
        x = self.bn4(x)
        x = F.softmax(self.linear2(x))
        return x, embedding

