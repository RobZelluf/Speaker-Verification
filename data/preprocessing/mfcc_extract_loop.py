import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import librosa
import soundfile as sf
import windowing as win
import warnings
warnings.filterwarnings("ignore")
import tkinter as tk
from tkinter import filedialog
from sklearn import preprocessing
toextract = ["26", "78", "118", "163", "196", "201", "254", "307", "311", "374", "405", "19", "39", "40", "83", "87", "89", "125", "150", "198", "200", "211"]
for i in toextract:
    directory = "/media/kaiser/NAS/LibriSpeech/train-clean-100/" + i
    mfcc = []
    mfcc_norm = []
    Fs = 16000
    frame_length = np.int(np.around(2 * Fs))  # 2s in samples
    window_types = ('rect', 'hann', 'cosine', 'hamming')
    frames = 0
    for dirpath, dirnames, files in os.walk(directory):
        print(f'Found directory: {dirpath}')
        for filename in files:
            if filename.endswith(".flac"):
                try:
                    y, sr = sf.read(dirpath + "/" + filename)

                    print("File read successfully:", dirpath + "/" + filename)
                    voice_matrix, number_of_frames = win.ex1_windowing(y, frame_length, window_types[0])  # Windowing

                    for i in range(number_of_frames):
                        frames += 1
                        #cur_frame = voice_matrix[:, i]
                        mfcc_per_frame = librosa.feature.mfcc(voice_matrix[:, i], sr=Fs, n_mfcc=30, hop_length=160)
                        mfcc.append(mfcc_per_frame)
                        mfcc_per_frame_norm = preprocessing.scale(mfcc_per_frame, axis=1)
                        mfcc_norm.append(preprocessing.scale(mfcc_per_frame, axis=1))
                        '''
                        plt.figure(1)
                        plt.imshow(mfcc_per_frame, origin='lower', aspect='auto')
                        plt.autoscale(enable=True, axis='both', tight=True)
                        plt.colorbar()
                        #plt.show()
    
                        plt.figure(2)
                        plt.imshow(mfcc_per_frame_norm, origin='lower', aspect='auto')
                        plt.autoscale(enable=True, axis='both', tight=True)
                        plt.colorbar()
                        plt.show()
                        '''
                except RuntimeError:
                    print("System Error:", filename)
    print(len(mfcc))

    #print(len(mfcc))
    #print((mfcc[0].shape))
    save_name = directory.split("/")[-1]
    pickle.dump(mfcc, open("../MFCC_Dumb/ID_" + save_name + ".p", "wb"))
    pickle.dump(mfcc_norm, open("../MFCC_Dumb/ID_" + save_name + "_norm.p", "wb"))
    print("Successfully saved to file: ID_{}.p".format(save_name))
    print("Used", frames * 2, "seconds in total")
    print("That equals to %.2f minutes" % (frames/30))