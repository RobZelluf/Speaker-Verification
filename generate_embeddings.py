import numpy as np
import pickle
from utils import *
import torch
from CNN import *
import os

DIR = input("Save in directory:")
if not os.path.exists("embeddings/" + DIR):
    os.mkdir("embeddings/" + DIR)

X, Y = read_data()

m = X.shape[0]
input_dim = [X.shape[1], X.shape[2]]
num_speakers = max(Y + 1)
X = X.reshape((m, 1, input_dim[0], input_dim[1]))

X = torch.from_numpy(X)
# Y = torch.from_numpy(Y)

model = torch.load("models/CNN1/CNN.pth")


for i in range(num_speakers):
    indices = []
    for j, ind in enumerate(Y):
        if ind == i:
            indices.append(j)

    embeddings = model(X[indices])
    print(embeddings.shape)

    with open("embeddings/" + DIR + "/" + str(i) + ".p", "wb") as f:
        pickle.dump(embeddings, f)



