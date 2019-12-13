import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, batch_size, output_dim=8, num_layers=1):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.RNN(self.input_dim[0], 64, self.num_layers)

        # setup output layer
        self.linear1 = nn.Linear(self.num_layers * 64, 700)
        self.linear2 = nn.Linear(700, output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        x = hidden.view(-1, (self.num_layers * 64))

        x = self.linear1(x)
        x = self.linear2(x)

        pred = F.softmax(x)
        return pred

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

