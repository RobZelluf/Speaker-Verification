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
if not os.path.exists("models/" + DIR + "/embeddings"):
    os.mkdir("models/" + DIR + "/embeddings")

model = torch.load("models/" + DIR + "/model.pth")
X, Y = read_data("data/processed/full_data.p")

# Load training indices and test indices
with open("models/" + DIR + "/train-test.p", "rb") as f:
    train_ind, test_ind = pickle.load(f)

print("Train samples:", len(train_ind))
print("Test_samples:", len(test_ind))

m = X.shape[0]
input_dim = [X.shape[1], X.shape[2]]
num_speakers = max(Y + 1)
X = X.reshape((m, input_dim[0], input_dim[1]))
X = torch.from_numpy(X)

for i in range(num_speakers):
    indices = []
    for j, ind in enumerate(Y):
        if ind == i:
            indices.append(j)

    _, embeddings = model(X[indices])

    with open("models/" + DIR + "/embeddings/" + str(i) + ".p", "wb") as f:
        pickle.dump(embeddings, f)



