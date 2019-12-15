import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, embedding_dim=700, num_layers=1):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.batch_size = input_dim[1]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.linear_size = hidden_dim * input_dim[1] * 2

        print("Num layers:", self.num_layers)
        print("Hidden_dim:", self.hidden_dim)
        print("Embedding_dim:", self.embedding_dim)

        # setup LSTM layer
        self.lstm = nn.LSTM(input_dim[0], self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # setup output layer
        self.linear1 = nn.Linear(self.linear_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, self.embedding_dim)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        self.linear3 = nn.Linear(self.embedding_dim, output_dim)

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.rand(self.num_layers * 2, batch_size, self.hidden_dim),
                  torch.rand(self.num_layers * 2, batch_size, self.hidden_dim))

        return hidden

    def forward(self, x):
        seq_len = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(seq_len)

        lstm_out, hidden = self.lstm(x, hidden)
        x = lstm_out.contiguous().view(-1, self.hidden_dim * self.input_dim[1] * 2)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = F.softmax(self.linear3(x))

        return x

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

