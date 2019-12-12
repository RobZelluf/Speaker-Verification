import numpy as np
from utils import read_data
from LSTM import *
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    print("Using GPU!")
    torch.cuda.set_device(0)

X, Y = read_data()
print(X.shape)
input_size = (X.shape[1], X.shape[2])
num_speakers = max(Y)
m = X.shape[0]

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

batch_size = 20

model = Model(input_size, 256, batch_size, num_speakers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

num_batches = int(m / batch_size)

iterations = 100000
for i in range(iterations):
    model.init_hidden()

    for i in range(num_batches):
        model.zero_grad()
        X_local_minibatch, y_local_minibatch = (
            X[i * batch_size: (i + 1) * batch_size, ],
            Y[i * batch_size: (i + 1) * batch_size, ],
        )

        X_local_minibatch = X_local_minibatch.permute(0, 2, 1)
        print(X_local_minibatch.shape)
        print(y_local_minibatch.shape)
        y_pred = model(X_local_minibatch)

        print(y_pred.shape)

        loss = criterion(y_pred, y_local_minibatch)
        loss.backward()
        optimizer.step()

print("Iteration {} out of {}. Loss: {}. Accuracy {}.".format(i, iterations, loss.detach(), model.get_accuracy(y_pred, Y)))

