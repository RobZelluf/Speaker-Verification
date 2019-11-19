import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, num_speakers):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.num_speakers = num_speakers

        self.lstm1 = nn.LSTM(input_dim, 256, 3, batch_first=True)
        self.linear1 = nn.Linear(1536, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 700)
        self.linear4 = nn.Linear(700, num_speakers)

    def forward(self, x):
        lstm_out, x = self.lstm1(x)
        x = torch.cat(x).flatten()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        x = F.avg_pool1d(x, 5)

        x = F.relu(self.linear3(x))
        x = F.normalize(x, 2, 1)
        x = F.softmax(x)
        return x

