import pickle
import os
import librosa
import soundfile as sf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn import preprocessing

def windowing(data, frame_length, windowing_function):
    data = np.array(data)
    number_of_frames = int(len(data) / 32000)
    frame_matrix = np.zeros((frame_length, number_of_frames))
    if windowing_function == 'rect':
        window = 1
    elif windowing_function == 'hann':
        window = np.hanning(frame_length)
    elif windowing_function == 'cosine':
        window = np.sqrt(np.hanning(frame_length))
    elif windowing_function == 'hamming':
        window = np.hamming(frame_length)
    else:
        os.error('Windowing function not supported')

    for i in range(number_of_frames):
        frame = np.zeros(frame_length)
        frame = window*data[i * frame_length:frame_length + i * frame_length]
        frame_matrix[:, i] = frame
    frame_matrix = np.asfortranarray(frame_matrix)
    return frame_matrix, number_of_frames

root = tk.Tk()
root.filename = filedialog.askdirectory(initialdir="/", title="Select folder")

mfcc = []
mfcc_norm = []
Fs = 16000
frame_length = np.int(np.around(2 * Fs))  # 2s in samples
window_types = ('rect', 'hann', 'cosine', 'hamming')
frames = 0
for dirpath, dirnames, files in os.walk(root.filename):
    print(f'Found directory: {dirpath}')
    for filename in files:
        if filename.endswith(".flac"):
            try:
                y, sr = sf.read(dirpath + "/" + filename)

                print("File read successfully:", dirpath + "/" + filename)
                voice_matrix, number_of_frames = windowing(y, frame_length, window_types[0])

                for i in range(number_of_frames):
                    frames += 1
                    mfcc_per_frame = librosa.feature.mfcc(voice_matrix[:, i], sr=Fs, n_mfcc=30, hop_length=160)
                    mfcc.append(mfcc_per_frame)
                    mfcc_per_frame_norm = preprocessing.scale(mfcc_per_frame, axis=1)
                    mfcc_norm.append(preprocessing.scale(mfcc_per_frame, axis=1))

            except RuntimeError:
                print("System Error:", filename)
print(len(mfcc))


save_name = directory.split("/")[-1]

pickle.dump(mfcc, open("../MFCC_Dumb/Speaker_ID_" + save_name + ".p", "wb"))
pickle.dump(mfcc_norm, open("../MFCC_Dumb/Speaker_ID_" + save_name + "_norm.p", "wb"))
print("Successfully saved to file: Speaker_ID_{}.p".format(save_name))
print("Used", frames * 2, "seconds in total")
print("That equals to %.2f minutes" % (frames / 30))