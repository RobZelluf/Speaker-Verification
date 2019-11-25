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
m = X.shape[0]
num_speakers = max(Y + 1)

X = torch.from_numpy(X)
input_dim = [X.shape[1], X.shape[2]]
X = X.reshape((m, 1, input_dim[0], input_dim[1]))

Y = torch.from_numpy(Y)

model = Model(input_dim, num_speakers)
optimizer = optim.Adam(model.parameters(), lr=0.001)

iterations = 1000
for i in range(iterations):
    y = model.forward(X)

    loss = torch.nn.CrossEntropyLoss()(y, Y)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    print("Iteration", i, "out of", iterations, ". Loss:", round(float(loss.detach()), 2), "- Accuracy", get_accuracy(y, Y))

