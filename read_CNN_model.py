import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

DIRs = os.listdir("models/")
for i in range(len(DIRs)):
    if DIRs[i][0] == "_":
        continue

    print(i, DIRs[i])

dir_ind = int(input("Model number:"))
DIR = DIRs[dir_ind]

model = torch.load("models/" + DIR + "/model.pth")

with open("models/" + DIR + "/performance.p", "rb") as f:
    test_accs = pickle.load(f)

print(test_accs)

plt.plot(test_accs)
plt.show()

