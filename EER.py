import os
import pickle
import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def gen_pairs(num_pairs):
    pairs = np.zeros((4*num_pairs*2,700))
    correct_pairs = np.zeros((2*num_pairs*2,700))
    incorrect_pairs = np.zeros((2*num_pairs*2,700))
    # Sample pairs
    for i in range(2):
        emb_path = os.getcwd() + "/embeddings/test2/"
        with open(emb_path + str(i) + ".p", "rb") as f:
            embeddings=pickle.load(f)
        emb_vecs = embeddings[1].detach().numpy()
        #np.random.shuffle(emb_vecs)
        correct_pairs[(2*num_pairs*i):(2*num_pairs*(i+1)),:] = emb_vecs[0:(2*num_pairs),:]
        incorrect_pairs[(2*num_pairs*i):(2*num_pairs*(i+1)),:] = emb_vecs[(2*num_pairs):(4*num_pairs),:]


    pairs[0:(2*num_pairs*2),:] = correct_pairs
    # Shuffle incorrect_pairs to make false matchings
    for i in range(2):
        for j in range(2*num_pairs):
            pairs_index = 2*num_pairs*2 + j*2 + i
            pairs[pairs_index,:] = incorrect_pairs[(i*2*num_pairs + j),:]
            print(pairs_index, " , ", i*2*num_pairs+j)
    return pairs

def compute_cosine_similarities(pairs):
    n_pairs = int(pairs.shape[0]/2)
    cos_sims = np.zeros(n_pairs)
    for i in range(n_pairs):
        vec1 = pairs[(2*i),:]
        vec2 = pairs[(2*i + 1),:]
        cos_sims[i] = cosine_similarity(vec1.reshape(1,-1),vec2.reshape(1,-1))
    return cos_sims

plot = 1
n_pairs = 100
pairs = gen_pairs(n_pairs)
cosine_similarities = compute_cosine_similarities(pairs)

true_labels = np.zeros(n_pairs*2*2)
true_labels[0:(n_pairs*2)] = np.ones(n_pairs*2)

# Compute EER
fp_rate, tp_rate, threshold = roc_curve(true_labels, cosine_similarities, pos_label=1)
fn_rate = 1 - tp_rate
EER = fp_rate[np.nanargmin(np.absolute((fn_rate - fp_rate)))]

print("EER: ", EER)

# Plot roc curve
if plot == 1:
    roc_auc = auc(fp_rate, tp_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fp_rate, tp_rate, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [1, 0],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
