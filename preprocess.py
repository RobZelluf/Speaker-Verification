import pickle as p
import os
import numpy as np

full_set = True

if full_set:
    DIR = "data/large_set/"
    print("Processing full set!")
else:
    DIR = "data/MFCC_Dumb/"
    print("Processing small set!")

X = []
Y = []

ID = 0
speaker_ids = dict()

for file in os.listdir(DIR):
    if not full_set:
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
X = np.array(X)
Y = np.array(Y)

if full_set:
    filename = "data/processed/full_data.p"
    filename2 = "data/processed/full_speaker_ids.p"
else:
    filename = "data/processed/data.p"
    filename2 = "data/processed/speaker_ids.p"

with open(filename, "wb") as f:
    data = [X, Y]
    p.dump(data, f)

with open(filename2, "wb") as f:
    p.dump(speaker_ids, f)

