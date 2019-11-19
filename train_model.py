import numpy as np
from utils import read_data
from Model import *
import torch


def get_accuracy(logits, target):
    """ compute accuracy for training round """
    corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
    ).sum()
    accuracy = 100.0 * corrects / len(target)
    return accuracy.item()


X, Y = read_data()
num_speakers = len(Y[0])

X = torch.from_numpy(X)
X = X.flatten(1, -1)
input_dim = X.shape[1]

X = X.reshape(1, X.shape[0], X.shape[1])
Y = torch.from_numpy(Y)

model = Model(input_dim, num_speakers)
optimizer = optim.Adam(model.parameters(), lr=0.001)

iterations = 1000
for i in range(iterations):
    y = model.forward(X)
    loss = torch.nn.NLLLoss()(y, Y)
    loss.backward()
    optimizer.step()
    print(loss.detach())
    print(get_accuracy(y, Y))

