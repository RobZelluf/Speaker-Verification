import numpy as np
from utils import read_data
from LSTM import *
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import math
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

DIR = "LSTM"
model_loaded = False

if not os.path.exists("models/" + DIR):
    os.mkdir("models/" + DIR)
else:
    model = torch.load("models/" + DIR + "/CNN.pth")
    model_loaded = True

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

input_dim = (X.shape[1], X.shape[2])
X = X.reshape((m, input_dim[1], input_dim[0]))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.from_numpy(X_train)

X_test = torch.from_numpy(X_test)
X_test = X_test.reshape((input_dim[1], X_test.shape[0], input_dim[0]))

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

m_train = X_train.shape[0]

batch_size = 128
batches = math.ceil(m_train / batch_size)

if not model_loaded:
    model = Model(input_dim, batch_size, num_speakers)
else:
    print("Not creating model, already loaded!")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()

epochs = 100000

train_accs = []
test_accs = []
for i in range(epochs):
    indices = list(range(m_train))
    random.shuffle(indices)
    total = []
    avg_acc = []
    avg_loss = []
    for j in range(batches):
        if j == batches - 1:
            continue

        start = j * batch_size
        batch_indices = indices[start:start + batch_size]
        total.extend(batch_indices)
        X_train_batch = X_train[batch_indices]
        y_train_batch = y_train[batch_indices]

        X_train_batch = X_train_batch.reshape((input_dim[1], X_train_batch.shape[0], input_dim[0]))
        y_pred = model(X_train_batch)

        avg_acc.append(get_accuracy(y_pred, y_train_batch))

        optimizer.zero_grad()
        loss = criterion(y_pred, y_train_batch)
        avg_loss.append(loss.detach())
        loss.backward(retain_graph=True)
        optimizer.step()

    y_test_pred = model(X_test)
    test_acc = get_accuracy(y_test_pred, y_test)
    print("Epoch {} out of {}. Loss: {:.3f}. Train-accuracy {:.3f}. Test-accuracy {:.3f}.".format(i, epochs, np.mean(avg_loss), np.mean(avg_acc), test_acc))

    test_accs.append(test_acc)

    torch.save(model, "models/" + DIR + "/CNN.pth")
    with open("models/" + DIR + "/performance.p", "wb") as f:
        pickle.dump(test_accs, f)
