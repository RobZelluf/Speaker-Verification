import pickle as p
import numpy as np


def read_data(filename="data/processed/data.p"):
    with open(filename, "rb") as f:
        X, Y = p.load(f)
        return X, Y
