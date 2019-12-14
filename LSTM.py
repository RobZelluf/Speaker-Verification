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

        print("Num layers:", self.num_layers)
        print("Hidden_dim:", self.hidden_dim)
        print("Embedding_dim:", self.embedding_dim)

        # setup LSTM layer
        self.lstm = nn.RNN(input_dim[0], self.hidden_dim, self.num_layers, batch_first=True)

        # setup output layer
        self.linear1 = nn.Linear(self.hidden_dim * self.input_dim[1], 700)
        self.linear2 = nn.Linear(700, output_dim)

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        lstm_out, hidden = self.lstm(x, hidden)
        x = lstm_out.contiguous().view(-1, self.hidden_dim * self.input_dim[1])

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

