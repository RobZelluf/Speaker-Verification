import torch
import numpy as np

model = torch.load("models/CNN.pth")

embeddings = model.linear2.weight.data.numpy()
embeddings = np.transpose(embeddings)

np.save("embeddings/CNN1.npy", embeddings)

