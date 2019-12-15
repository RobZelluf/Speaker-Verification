import pickle as p
import os
import numpy as np

DIR = "data/MFCC_Dumb/"

X = []
Y = []

ID = 0
speaker_ids = dict()

for file in os.listdir(DIR):
    if "norm" not in file:
        continue

    speaker_ids[file] = ID

    file = DIR + file
    if os.path.isdir(file):
        continue

    with open(file, 'rb') as f:
        data = p.load(f)
        num_fragments = len(data)
        X.extend(data)
        labels = [ID] * num_fragments
        Y.extend(labels)

    ID += 1

print("Num speakers", ID)
exit()
X = np.array(X)
Y = np.array(Y)


with open("data/processed/data.p", "wb") as f:
    data = [X, Y]
    p.dump(data, f)

with open("data/processed/speaker_ids.p", "wb") as f:
    p.dump(speaker_ids, f)

