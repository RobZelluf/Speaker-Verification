import numpy as np
from utils import read_data
from CNN import *
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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

input_dim = [X.shape[1], X.shape[2]]
X = X.reshape((m, 1, input_dim[0], input_dim[1]))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

m_train = X_train.shape[0]

model = Model(num_speakers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()

batch_size = 100

iterations = 100000
for i in range(iterations):
    batch_mask = np.random.choice(m_train - 1, batch_size)
    X_train_batch = X_train[batch_mask]
    y_train_batch = y_train[batch_mask]

    print(X_train_batch)

    y_pred = model(X_train_batch)

    optimizer.zero_grad()
    loss = criterion(y_pred, y_train_batch)
    loss.backward()
    optimizer.step()

    print("Iteration {} out of {}. Loss: {:.3f}. Accuracy {:.3f}.".format(i, iterations, loss.detach(), get_accuracy(y_pred, y_train_batch)))

