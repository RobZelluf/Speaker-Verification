import numpy as np
from utils import read_data
from Model import *
import torch

X, Y = read_data()

X = torch.from_numpy(X)
X = X.flatten(1, -1)
input_dim = X.shape[1]

X = X.reshape(1, X.shape[0], X.shape[1])
Y = torch.from_numpy(Y)

num_speakers = torch.max(Y) + 1

model = Model(input_dim, num_speakers)

y = model.forward(X)

