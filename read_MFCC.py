import pickle as p

DIR = "data/MFCC_Dumb/"
filename = "ID_19.p"

with open(DIR + filename, 'rb') as f:
    data = p.load(f)