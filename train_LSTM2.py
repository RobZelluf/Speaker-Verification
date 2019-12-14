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
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--embedding_size", type=int, default=700)
args = parser.parse_args()


DIR = input("Model directory:")
model_loaded = False

if not os.path.exists("models/" + DIR):
    os.mkdir("models/" + DIR)
    with open("models/" + DIR + "/model.txt", "w") as f:
        f.write("Num layers " + str(args.num_layers) + "\n")
        f.write("Hidden size " + str(args.hidden_size) + "\n")
        f.write("Embedding size " + str(args.embedding_size) + "\n")
else:
    model = torch.load("models/" + DIR + "/model.pth")
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
indices = list(range(len(X)))

if "train-test.p" not in os.listdir("models/" + DIR):
    X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(X, Y, indices, test_size=0.2)
    with open("models/" + DIR + "/train-test.p", "wb") as f:
        pickle.dump([train_ind, test_ind], f)
        print("Train/test data saved!")
else:
    with open("models/" + DIR + "/train-test.p", "rb") as f:
        train_ind, test_ind = pickle.load(f)
        print("Train-test data loaded!")

    X_train = X[train_ind]
    X_test = X[test_ind]
    y_train = Y[train_ind]
    y_test = Y[test_ind]


X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
X_test = X_test.reshape((input_dim[1], X_test.shape[0], input_dim[0]))

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

m_train = X_train.shape[0]

batch_size = 128
batches = math.ceil(m_train / batch_size)

if not model_loaded:
    model = Model(input_dim, num_speakers, args.hidden_size, args.embedding_size, args.num_layers)
    torch.save(model, "models/" + DIR + "/model.pth")
else:
    print("Not creating model, already loaded!")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

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
        # if j == batches - 1:
        #     continue

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

    perf_string = "Epoch {} out of {}. Loss: {:.3f}. Train-accuracy {:.3f}. Test-accuracy {:.3f}.".format(i, epochs, np.mean(avg_loss), avg_acc, test_acc)
    print(perf_string)
    with open("models/" + DIR + "/performance.txt", "a") as f:
        f.write(perf_string)
        f.write("\n")

    test_accs.append([avg_acc, test_acc])

    torch.save(model, "models/" + DIR + "/model.pth")
    with open("models/" + DIR + "/performance.p", "wb") as f:
        pickle.dump(test_accs, f)
