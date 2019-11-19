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

    def forward(self, x):
        x = self.lstm1(x)
        return x
