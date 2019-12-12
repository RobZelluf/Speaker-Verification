import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

model = torch.load("models/CNN1/CNN.pth")

embeddings = model.linear2.weight.data.numpy()
embeddings = np.transpose(embeddings)

np.save("embeddings/CNN1.npy", embeddings)

with open("models/CNN1/performance.p", "rb") as f:
    train_accs, test_accs = pickle.load(f)

plt.plot(train_accs)
plt.plot(test_accs)
plt.show()

