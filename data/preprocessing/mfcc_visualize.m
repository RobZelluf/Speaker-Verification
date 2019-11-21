[y, fs] = audioread('C:\Users\Franz Kaiser\Downloads\SX83.wav');
mfccs = mfcc(y, fs);
imagesc(mfccs)
