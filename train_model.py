import numpy as np
from utils import read_data
from CNN import *
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    print("Using GPU!")
    torch.cuda.set_device(0)


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

model = Model(num_speakers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()

batch_size = 100

iterations = 100000
for i in range(iterations):
    train_mask = np.random.choice(m - 1, batch_size)
    X_train = X[train_mask]
    Y_train = Y[train_mask]

    y_pred = model(X_train)

    optimizer.zero_grad()
    loss = criterion(y_pred, Y_train)
    loss.backward()
    optimizer.step()

    y_pred = model(X)
    print("Iteration {} out of {}. Loss: {:.3f}. Accuracy {:.3f}.".format(i, iterations, loss.detach(), get_accuracy(y_pred, Y)))

