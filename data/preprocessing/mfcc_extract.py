import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import librosa
import soundfile as sf
import windowing as win
import tkinter as tk
from tkinter import filedialog
from sklearn import preprocessing

root = tk.Tk()
root.filename = filedialog.askdirectory(initialdir="/media/kaiser/NAS/LibriSpeech/train-clean-100", title="Select folder")
#directory = "/media/kaiser/NAS/LibriSpeech/train-clean-100/118"
mfcc = []
mfcc_norm = []
Fs = 16000
frame_length = np.int(np.around(2 * Fs))  # 2s in samples
window_types = ('rect', 'hann', 'cosine', 'hamming')
frames = 0
for dirpath, dirnames, files in os.walk(root.filename):
    print(f'Found directory: {dirpath}')
    for filename in files:
        if filename.endswith(".wav"):
            try:
                y, sr = sf.read(dirpath + "/" + filename)

                print("File read successfully:", dirpath + "/" + filename)
                voice_matrix, number_of_frames = win.windowing(y, frame_length, window_types[0])  # Windowing

                for i in range(number_of_frames):
                    frames += 1
                    #cur_frame = voice_matrix[:, i]
                    mfcc_per_frame = librosa.feature.mfcc(voice_matrix[:, i], sr=Fs, n_mfcc=30, hop_length=160)
                    mfcc.append(mfcc_per_frame)
                    mfcc_per_frame_norm = preprocessing.scale(mfcc_per_frame, axis=1)
                    mfcc_norm.append(preprocessing.scale(mfcc_per_frame, axis=1))

                    plt.figure(1)
                    COLOR = 'white'
                    plt.rcParams['text.color'] = COLOR
                    plt.rcParams['axes.labelcolor'] = COLOR
                    plt.rcParams['xtick.color'] = COLOR
                    plt.rcParams['ytick.color'] = COLOR
                    plt.figure().patch.set_facecolor('xkcd:black')
                    plt.subplot(1,2,1)
                    plt.imshow(mfcc_per_frame, origin='lower', aspect='auto')
                    plt.autoscale(enable=True, axis='both', tight=True)
                    plt.colorbar()
                    size = 18
                    plt.xlabel("Frame number", fontsize=size)
                    plt.ylabel("MFCC number", fontsize=size)
                    plt.title("Not normalized", fontsize=size)
                    #plt.show()

                    plt.subplot(1,2,2)
                    plt.imshow(mfcc_per_frame_norm, origin='lower', aspect='auto')
                    plt.autoscale(enable=True, axis='both', tight=True)
                    plt.colorbar()
                    plt.xlabel("Frame number", fontsize=size)
                    plt.ylabel("MFCC number", fontsize=size)
                    plt.title("Normalized", fontsize=size)
                    plt.show()
                    #plt.savefig("mfcc_plot.png", dpi=600, facecolor='black', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None, metadata=None)

            except RuntimeError:
                print("System Error:", filename)
print(len(mfcc))

#print(len(mfcc))
#print((mfcc[0].shape))
#save_name = directory.split("/")[-1]

#pickle.dump(mfcc, open("../MFCC_Dumb/Speaker_ID_" + save_name + ".p", "wb"))
#pickle.dump(mfcc_norm, open("../MFCC_Dumb/Speaker_ID_" + save_name + "_norm.p", "wb"))
#print("Successfully saved to file: Speaker_ID_{}.p".format(save_name))
#print("Used", frames * 2, "seconds in total")
#print("That equals to %.2f minutes" % (frames / 30))