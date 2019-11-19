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
        self.linear1 = nn.Linear(256, num_speakers)

    def forward(self, x):
        lstm_out, hidden = self.lstm1(x)
        logits = self.linear1(lstm_out[-1])
        speaker_scores = F.log_softmax(logits, dim=1)
        return speaker_scores
