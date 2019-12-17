import numpy as np
import pickle
from utils import *
import torch
from LSTM import *
import os

DIRs = os.listdir("models/")
for i in range(len(DIRs)):
    if DIRs[i][0] == "_":
        continue

    print(i, DIRs[i])

dir_ind = int(input("Model number:"))
DIR = DIRs[dir_ind]

model = torch.load("models/" + DIR + "/model.pth")
X, Y = read_data("data/processed/full_data.p")

# Load training indices and test indices
with open("models/" + DIR + "/train-test.p", "rb") as f:
    train_ind, test_ind = pickle.load(f)

print("Train samples:", len(train_ind))
print("Test_samples:", len(test_ind))

X = X[test_ind]
Y = Y[test_ind]

m = X.shape[0]
input_dim = [X.shape[1], X.shape[2]]
num_speakers = max(Y + 1)

print("Number of samples:", m)
print("Number of speakers:", num_speakers)

X = X.reshape((m, input_dim[1], input_dim[0]))
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

accuracies = []
for i in range(num_speakers):
    indices = []
    for j, ind in enumerate(Y):
        if ind == i:
            indices.append(j)

    y_pred, _ = model(X[indices])

    test_acc = get_accuracy(y_pred, Y[indices])
    accuracies.append(test_acc)

print("Accuracy:", np.mean(accuracies))



