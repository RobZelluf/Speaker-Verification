import numpy as np
from utils import read_data
from CNN import *
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import math
import pickle

import warnings
warnings.filterwarnings("ignore")

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.NLLLoss()

batch_size = 500
batches = math.ceil(m_train / batch_size)

epochs = 100000

train_accs = []
test_accs = []
for i in range(epochs):
    indices = list(range(m_train))
    random.shuffle(indices)
    total = []
    for j in range(batches):
        start = j * batch_size
        batch_indices = indices[start:start + batch_size]
        total.extend(batch_indices)
        X_train_batch = X_train[batch_indices]
        y_train_batch = y_train[batch_indices]

        y_pred = model(X_train_batch)

        optimizer.zero_grad()
        loss = criterion(y_pred, y_train_batch)
        loss.backward()
        optimizer.step()

    y_test_pred = model(X_test)
    test_acc = get_accuracy(y_test_pred, y_test)
    print("Epoch {} out of {}. Loss: {:.3f}. Accuracy {:.3f}.".format(i, epochs, loss.detach(), test_acc))

    y_train_pred = model(X_train)
    train_acc = get_accuracy(y_train_pred, y_train)
    print("Train accuracy {:.3f}".format(train_acc))

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    torch.save(model, "models/CNN1/CNN.pth")
    with open("models/CNN1/performance.p", "wb") as f:
        pickle.dump([train_accs, test_accs], f)
