import pickle as p
import numpy as np
import torch


def read_data(filename="data/processed/data.p"):
    with open(filename, "rb") as f:
        X, Y = p.load(f)
        return X, Y


def get_accuracy(logits, target):
    """ compute accuracy for training round """
    corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
    ).sum()
    accuracy = 100.0 * corrects / len(target)
    return accuracy.item()
