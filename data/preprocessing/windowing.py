import os
import numpy as np


def ex1_windowing(data, frame_length, windowing_function):
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

    #print(len(data))
    #mod = len(data) % frame_length
    #np.pad(data, (0, frame_length - mod), 'constant', constant_values=(0, 0))
    for i in range(number_of_frames):
        frame = np.zeros(frame_length)
        frame = window*data[i * frame_length:frame_length + i * frame_length]
        frame_matrix[:, i] = frame
    frame_matrix = np.asfortranarray(frame_matrix)
    #print("Done windowing")
    return frame_matrix, number_of_frames
