import numpy as np
import pickle
from utils import *
import torch
from CNN1D import *
import os

DIRs = os.listdir("models/")
for i in range(len(DIRs)):
    if DIRs[i][0] == "_":
        continue

    print(i, DIRs[i])

dir_ind = int(input("Model number:"))
DIR = DIRs[dir_ind]

# Create folder for embeddings if it does not yet exist
if not os.path.exists("models/" + DIR + "/embeddings_test_normalized"):
    os.mkdir("models/" + DIR + "/embeddings_test_normalized")

model = torch.load("models/" + DIR + "/model.pth")
model.eval()

X, Y = read_data("data/processed/full_data.p")

# Load training indices and test indices
with open("models/" + DIR + "/train-test.p", "rb") as f:
    train_ind, test_ind = pickle.load(f)

test = False

if test:
    X = X[test_ind]
    Y = Y[test_ind]
else:
    cutoff = np.where(Y == 100)[0][0]
    X = X[:cutoff]
    Y = Y[:cutoff]

m = X.shape[0]
input_dim = [X.shape[1], X.shape[2]]
num_speakers = max(Y + 1)

print("Training examples:", m)
print("Number of speakers:", num_speakers)

X = X.reshape((m, input_dim[0], input_dim[1]))
X = torch.from_numpy(X)

print("X", X.shape)
print("Y", Y.shape)

accuracies = []
for i in range(num_speakers):
    indices = []
    for j, ind in enumerate(Y):
        if ind == i:
            indices.append(j)

    y_pred, embeddings = model(X[indices])
    accuracies.append(get_accuracy(y_pred, torch.from_numpy(Y[indices])))
    print("Accuracy:", np.mean(accuracies))

    with open("models/" + DIR + "/embeddings_test_normalized/" + str(i) + ".p", "wb") as f:
        pickle.dump(embeddings, f)



