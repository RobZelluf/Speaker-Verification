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
    accuracies = pickle.load(f)

print("Train accuracy:", round(accuracies[-1][0], 2))
print("Test accuracy:", round(accuracies[-1][1], 2))

plt.plot(accuracies)
plt.legend(["Train accuracy", "Test accuracy"])
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy (%)")
plt.show()

