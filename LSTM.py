import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, batch_size, output_dim=8, hidden_dim = 64, num_layers=2):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        print(self.input_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.input_dim[0], self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        logits = self.linear(lstm_out[-1])
        pred = F.log_softmax(logits, dim=1)
        return pred

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

